
import os
from typing import Tuple, Union, List
import numpy as np
import torch
from torch import nn
from torch import autocast
from torch import distributed as dist
from sklearn.metrics import f1_score, confusion_matrix
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
import torch.nn.functional as F

from torch.nn import init

class CA_Module_3D(nn.Module):
    """3D Channel Attention (Modified from your 2D version)"""
    def __init__(self, in_channel):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)  # Changed to 3D
        self.maxpool = nn.AdaptiveMaxPool3d(1)   # Changed to 3D
        
        self.linear = nn.Sequential(
            nn.Linear(2 * in_channel, max(4, in_channel // 16)),  # Avoid too small dims
            nn.ReLU(),
            nn.Linear(max(4, in_channel // 16), in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()  # 3D shape
        p1 = self.avgpool(x).flatten(1)
        p2 = self.maxpool(x).flatten(1)
        p = torch.cat([p1, p2], dim=1)
        po = self.linear(p).view(b, c, 1, 1, 1)  # 3D reshape
        return x * po  # Remove ReLU to prevent dead neurons


class Classifier3D(nn.Module):
    """3D Classifier for nnUNet Features"""
    def __init__(self, num_classes, encoder_channels=[256, 320, 320]):
        super().__init__()
        # Channel attention for multi-scale features
        self.ca1 = CA_Module_3D(encoder_channels[0])
        self.ca2 = CA_Module_3D(encoder_channels[1])
        self.ca3 = CA_Module_3D(encoder_channels[2])
        
        # 3D Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        total_channels = sum(encoder_channels)
        self.fc = nn.Sequential(
            nn.Linear(total_channels, 128),  # Reduced capacity for 3D
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x1, x2, x3):
        # Process multi-scale features
        c1 = self.ca1(x1)  # [B, 256, D1, H1, W1]
        c2 = self.up(self.ca2(x2))  # [B, 320, D1, H1, W1]
        c3 = self.ca3(x3)  # [B, 320, D3, H3, W3]
        
        # Ensure spatial dims match via adaptive pooling
        target_size = c1.shape[2:]
        c2 = F.interpolate(c2, size=target_size, mode='trilinear')
        c3 = F.interpolate(c3, size=target_size, mode='trilinear')
        
        # Concatenate and classify
        x = torch.cat([c1, c2, c3], dim=1)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class nnUNetTrainer_CLSHeadCA(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mt_num_classes = int(os.environ.get('NNUNETV2_MT_NUM_CLS', '2'))
        self.mt_loss_weight = float(os.environ.get('NNUNETV2_MT_LOSS_WEIGHT', '0.3'))
        self.mt_multilabel = os.environ.get('NNUNETV2_MT_MULTILABEL', '0').lower() in ('1', 'true', 't', 'yes')
        
        # Calculate proper class weights based on your training data
        self.cls_head = None
        self.cls_loss_fn = None
        # self.gradient_accumulation_steps = 4  # Effective batch_size=8

    def _build_cls_head_if_needed(self):
        if self.cls_head is not None:
            return
            
        # encoder_channels = 320  # From plans.json
        
        # # Improved classification head
        # # self.cls_head = nn.Sequential(
        # #     nn.AdaptiveAvgPool3d(1),
        # #     nn.Flatten(),
        # #     nn.InstanceNorm1d(encoder_channels),  # Stable normalization for small batches
        # #     nn.Linear(encoder_channels, self.mt_num_classes)  # Direct prediction
        # # ).to(self.device)
        # self.cls_head = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),  
        #     nn.Flatten(),
        #     nn.LayerNorm(encoder_channels),
        #     nn.Linear(encoder_channels, self.mt_num_classes)
        # ).to(self.device) # simple LNorm experiment/ 1opt1lr experiment / simple ver2
        self.cls_head = Classifier3D(self.mt_num_classes).to(self.device)

        # Proper initialization
        # for m in self.cls_head.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #         nn.init.constant_(m.bias, 0)

        weights_fn = torch.tensor([1.7, 1.0, 1.26], dtype=torch.float32).to(self.device)  # Example weights
        
        # Improved loss function with label smoothing
        self.cls_loss_fn = nn.CrossEntropyLoss(
            weight=weights_fn,
            label_smoothing=0.1  # Helps with small batches
        )

    def configure_optimizers(self):
        self._build_cls_head_if_needed()
        # optimizer = torch.optim.SGD([
        #     {'params': self.network.parameters(), 'lr': self.initial_lr},
        #     {'params': self.cls_head.parameters(), 'lr': self.initial_lr * 5}  # Higher LR for head
        # ], lr=self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)

        optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.cls_head.parameters()),
            self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
        )
        
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def _compute_segmentation_loss_only(self, output, target):
        return self.loss(output, target)

    def _compute_classification_logits(self, encoder_features: torch.Tensor) -> torch.Tensor:
        # Use encoder features (320 channels) as input to the classification head
        # Ensure encoder features are float32 for the classification head
        if encoder_features.dtype != torch.float32:
            encoder_features = encoder_features.float()
        return self.cls_head(encoder_features)


    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device)
        target = batch['target']
        cls_target = batch.get('cls_target')
        
        if isinstance(target, list):
            target = [i.to(self.device) for i in target]
        else:
            target = target.to(self.device)
            
        if cls_target is not None:
            cls_target = cls_target.to(self.device).long()

        # Forward pass
        with autocast(self.device.type, enabled=True):
            # Single forward pass through encoder
            encoder_features = self.network.encoder(data)
            seg_outputs = self.network.decoder(encoder_features)
            bottleneck_feature_0 = encoder_features[-3]
            bottleneck_feature_1 = encoder_features[-2]
            bottleneck_feature_2 = encoder_features[-1]
            
            seg_loss = self._compute_segmentation_loss_only(seg_outputs, target)
            
            cls_loss = torch.tensor(0.0, device=self.device)
            if cls_target is not None:
                cls_logits = self.cls_head(bottleneck_feature_0, bottleneck_feature_1, bottleneck_feature_2)
                cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            total_loss = 0.7 * seg_loss + 0.3 * cls_loss
            # total_loss = 0.7 * seg_loss + self.mt_loss_weight * cls_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.network.parameters()) + list(self.cls_head.parameters()), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.network.parameters()) + list(self.cls_head.parameters()), 12)
            self.optimizer.step()

        # Metrics
        out = {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'cls_loss': cls_loss.detach().cpu().numpy() if cls_target is not None else 0
        }
        
        return out

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        cls_target = batch.get('cls_target', None)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        if cls_target is not None:
            if isinstance(cls_target, torch.Tensor):
                cls_target = cls_target.to(self.device, non_blocking=True)
            else:
                cls_target = torch.as_tensor(cls_target, device=self.device)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Forward pass - now returns seg_outputs AND encoder_features
            encoder_features = self.network.encoder(data)
            seg_outputs = self.network.decoder(encoder_features)
            bottleneck_feature_0 = encoder_features[-3]
            bottleneck_feature_1 = encoder_features[-2]
            bottleneck_feature_2 = encoder_features[-1]
            

            seg_loss = self._compute_segmentation_loss_only(seg_outputs, target)

            cls_loss = torch.tensor(0.0, device=self.device)
            if cls_target is not None and encoder_features is not None:
                cls_logits = self.cls_head(bottleneck_feature_0, bottleneck_feature_1, bottleneck_feature_2)

                cls_target = cls_target.long()
                cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            # total_loss = seg_loss + self.mt_loss_weight * cls_loss
            total_loss = 0.7 * seg_loss + 0.3 * cls_loss

        # Compute segmentation metrics (pseudo dice) as in base class
        if self.enable_deep_supervision:
            output = seg_outputs[0]
            target_for_metrics = target[0]
        else:
            output = seg_outputs
            target_for_metrics = target

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_metrics != self.label_manager.ignore_label).float()
                target_for_metrics[target_for_metrics == self.label_manager.ignore_label] = 0
            else:
                if target_for_metrics.dtype == torch.bool:
                    mask = ~target_for_metrics[:, -1:]
                else:
                    mask = 1 - target_for_metrics[:, -1:]
                target_for_metrics = target_for_metrics[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_for_metrics, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        out = {'loss': total_loss.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
        out['seg_loss'] = seg_loss.detach().cpu().numpy()
        out['cls_loss'] = cls_loss.detach().cpu().numpy()
        
        # Calculate F1 score for classification if we have classification targets
        if cls_target is not None and encoder_features is not None:
            cls_pred = cls_logits.argmax(dim=1)
            cls_pred_np = cls_pred.detach().cpu().numpy()
            cls_target_np = cls_target.detach().cpu().numpy()
            
            # Calculate metrics
            out['f1_macro'] = f1_score(cls_target_np, cls_pred_np, average='macro', zero_division=0)
            # cm = confusion_matrix(cls_target_np, cls_pred_np)
            
            # # Log per-class accuracy
            # for i in range(self.mt_num_classes):
            #     out[f'cls_acc_{i}'] = (cm[i,i] / cm[i].sum()) if cm[i].sum() > 0 else 0
        return out

    def on_epoch_end(self):
        super().on_epoch_end()
        
        # Only log if we're the main process (DDP) and have classification data
        if self.local_rank == 0 and hasattr(self.logger, 'my_fantastic_logging'):
            # Log classification metrics if they exist
            if 'f1_macro' in self.logger.my_fantastic_logging:
                last_f1 = self.logger.my_fantastic_logging['f1_macro'][-1]
                self.print_to_log_file(f"Validation F1 Macro: {np.round(last_f1, decimals=4)}")
                
                # Log per-class accuracy if available
                for i in range(self.mt_num_classes):
                    acc_key = f'cls_acc_{i}'
                    if acc_key in self.logger.my_fantastic_logging:
                        last_acc = self.logger.my_fantastic_logging[acc_key][-1]
                        self.print_to_log_file(f"Class {i} Accuracy: {np.round(last_acc * 100, decimals=2)}%")

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                # Get state dict from the original network (unwrapped)
                network_state = self.network.original_network.state_dict() if hasattr(self.network, 'original_network') else self.network.state_dict()
                
                checkpoint = {
                    'network_weights': (network_state if not self.is_ddp else self.network.module.original_network.state_dict()),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'cls_head_state': self.cls_head.state_dict() if self.cls_head is not None else None,
                    'mt_num_classes': self.mt_num_classes,
                    'mt_loss_weight': self.mt_loss_weight,
                    'mt_multilabel': self.mt_multilabel,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
            
        # Load network weights into the original network
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.original_network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)

        # Load into original network
        self.network.original_network.load_state_dict(new_state_dict)
        
        # Build and load classification head
        self._build_cls_head_if_needed()
        if checkpoint.get('cls_head_state') is not None:
            self.cls_head.load_state_dict(checkpoint['cls_head_state'])

        # Rebuild optimizer to include cls_head parameters
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])