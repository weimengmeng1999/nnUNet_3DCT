import os
from typing import Tuple, Union, List

import numpy as np
import torch
from torch import nn
from torch import autocast
from torch import distributed as dist
from sklearn.metrics import f1_score

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context

from torch.optim.lr_scheduler import CosineAnnealingLR
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


class nnUNetTrainer_CLSHead_Frozen(nnUNetTrainer):
    """
    Multi-task trainer that freezes the pre-trained encoder and only fine-tunes 
    a case-level classification head using encoder features.
    
    The classification head consumes the encoder's bottleneck features via
    AdaptiveAvgPool3d -> Flatten -> Dropout -> Linear.
    
    The encoder weights are frozen and only the classification head parameters
    are updated during training.

    Configuration via environment variables:
    - NNUNETV2_MT_NUM_CLS: number of classification classes (default: 2)
    - NNUNETV2_MT_LOSS_WEIGHT: lambda weight for classification loss (default: 0.3)
    - NNUNETV2_MT_MULTILABEL: '1' or 'true' for multilabel (BCEWithLogits), else multiclass (CrossEntropy)
    - NNUNETV2_MT_FREEZE_ENCODER: '1' or 'true' to freeze encoder (default: True)
    - NNUNETV2_PRE_CHECKPOINT_PATH: pretrained model path for loading checkpoints (optional)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mt_num_classes: int = int(os.environ.get('NNUNETV2_MT_NUM_CLS', '2'))
        self.mt_loss_weight: float = float(os.environ.get('NNUNETV2_MT_LOSS_WEIGHT', '0.3'))
        self.mt_multilabel: bool = os.environ.get('NNUNETV2_MT_MULTILABEL', '0').lower() in ('1', 'true', 't', 'yes')
        self.mt_freeze_encoder: bool = os.environ.get('NNUNETV2_MT_FREEZE_ENCODER', '1').lower() in ('1', 'true', 't', 'yes')
        
        # Custom checkpoint path
        self.pre_checkpoint_path = os.environ.get('NNUNETV2_PRE_CHECKPOINT_PATH', None)

        self.cls_head: nn.Module = None
        self.cls_loss_fn: nn.Module = None


    def _freeze_encoder_parameters(self):
        """Freeze all encoder parameters to prevent gradient updates"""
        if self.mt_freeze_encoder:
            for name, param in self.network.named_parameters():
                param.requires_grad = False
        else:
            print("Warning: Encoder parameters are NOT frozen!")

    def _build_cls_head_if_needed(self):
        if self.cls_head is not None:
            return
        
        # this is hardcoded for now
        # Input channels for the head come from the encoder (320 channels based on our plan.json for 3d fullers)
        # change this based on the last feature size of the encoder in plans.json
        encoder_channels = 320
        
        print(f"Building classification head with {encoder_channels} input channels (encoder features)")
        
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  
            nn.Flatten(),
            nn.LayerNorm(encoder_channels),
            nn.Linear(encoder_channels, self.mt_num_classes)
        ).to(self.device) # simple LNorm experiment
        # self.cls_head = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(encoder_channels, 256),
        #     # nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(256, self.mt_num_classes)
        # ).to(self.device) # for NBND experiments


        weights_fn = torch.tensor([1.3548, 0.7925, 1.0], dtype=torch.float32).to(self.device)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=weights_fn)
        # self.cls_loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        # Called in initialize() AND we call it again after attaching cls_head in on_train_start.
        # Ensure head exists before creating optimizer so its params are included.
        self._build_cls_head_if_needed()
        
        # Only optimize classification head parameters since encoder is frozen
        # trainable_params = list(self.cls_head.parameters())
        
        # Use higher learning rate for classification head since encoder is frozen
        # optimizer = torch.optim.SGD(
        #     self.cls_head.parameters(),
        #     lr=1e-4,  # 5× higher than base (if base was 0.01)
        #     weight_decay=self.weight_decay,
        #     momentum=0.99,
        #     nesterov=True
        # )
        optimizer = torch.optim.AdamW(
                    self.cls_head.parameters(),
                    lr=3e-3,          # slightly higher than before
                    weight_decay=1e-4
            )
        
        # from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        # lr_scheduler = PolyLRScheduler(optimizer, 1e-4, self.num_epochs)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return optimizer, lr_scheduler

    def on_train_start(self):
        super().on_train_start()
        # Rebuild optimizer to include classification head
        self._build_cls_head_if_needed()
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

    def _compute_segmentation_loss_only(self, output, target):
        return self.loss(output, target)

    def _compute_classification_logits(self, encoder_features: torch.Tensor) -> torch.Tensor:
        # Use encoder features (320 channels) as input to the classification head
        # Ensure encoder features are float32 for the classification head
        if encoder_features.dtype != torch.float32:
            encoder_features = encoder_features.float()
        return self.cls_head(encoder_features)

    def train_step(self, batch: dict) -> dict:
        # Load pretrained checkpoint if specified and not already loaded
        if self.pre_checkpoint_path is not None and not hasattr(self, '_pretrained_loaded'):
            print(f"Loading pretrained weights from: {self.pre_checkpoint_path}")
            checkpoint = torch.load(self.pre_checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load network weights into the original network
            new_state_dict = {}
            for k, value in checkpoint['network_weights'].items():
                key = k
                if key not in self.network.state_dict().keys() and key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value
            
            # Load the weights
            self.network.load_state_dict(new_state_dict)
            
            # Mark as loaded to avoid reloading every step
            self._pretrained_loaded = True
            print("Pretrained weights loaded successfully!")
            
            # Ensure encoder remains frozen after loading
            self._freeze_encoder_parameters()

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
                # convert numpy/array-like to tensor
                cls_target = torch.as_tensor(cls_target, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Forward pass - now returns seg_outputs AND encoder_features
            with torch.no_grad():
                encoder_features = self.network.encoder(data)[-1]

            cls_loss = torch.tensor(0.0, device=self.device)
            if cls_target is not None and encoder_features is not None:
                # print(f"Encoder features shape: {encoder_features.shape}")
                
                cls_logits = self._compute_classification_logits(encoder_features)
                
                cls_target = cls_target.long()
                # print('logits', cls_logits)
                # print('target',cls_target)
                cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            total_loss = cls_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # Only clip gradients for classification head since encoder is frozen
            torch.nn.utils.clip_grad_norm_(list(self.cls_head.parameters()), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            # Only clip gradients for classification head since encoder is frozen
            torch.nn.utils.clip_grad_norm_(list(self.cls_head.parameters()), 12)
            self.optimizer.step()

        # Report total loss; optionally also return cls/seg components for debugging
        out = {'loss': total_loss.detach().cpu().numpy()}
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
            seg_outputs = self.network(data)
            encoder_features = self.network.encoder(data)[-1]

            cls_loss = torch.tensor(0.0, device=self.device)
            if cls_target is not None and encoder_features is not None:
                cls_logits = self._compute_classification_logits(encoder_features)
                if self.mt_multilabel:
                    if cls_target.dtype != torch.float32:
                        cls_target = cls_target.float()
                    cls_loss = self.cls_loss_fn(cls_logits, cls_target)
                else:
                    cls_target = cls_target.long()
                    cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            total_loss = cls_loss

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
        
        # Calculate F1 score for classification if we have classification targets
        if cls_target is not None and encoder_features is not None:
            cls_logits = self._compute_classification_logits(encoder_features)
            # For single label multiclass, use argmax
            cls_pred = cls_logits.argmax(dim=1)
            # Convert to numpy for sklearn
            cls_pred_np = cls_pred.detach().cpu().numpy()
            cls_target_np = cls_target.detach().cpu().numpy()
            # Calculate macro F1 score
            f1_macro = f1_score(cls_target_np, cls_pred_np, average='macro', zero_division=0)
            
            print(f"F1 Macro: {f1_macro:.4f}")
        
        return out

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                # Get state dict from the original network (unwrapped)
                network_state = self.network.state_dict() if hasattr(self.network, 'original_network') else self.network.state_dict()
                
                checkpoint = {
                    'network_weights': (network_state if not self.is_ddp else self.network.module.state_dict()),
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
                    'mt_freeze_encoder': self.mt_freeze_encoder,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        # # Use custom checkpoint path if specified
        # if self.pre_checkpoint_path is not None:
            # Extract just the filename from the original path
        print(f"Loading checkpoint from pretrained path: {self.pre_checkpoint_path}")
        checkpoint = torch.load(self.pre_checkpoint_path, map_location=self.device, weights_only=False)
            
        # Load network weights into the original network
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)

        # Load into original network
        self.network.load_state_dict(new_state_dict)
        
        # Build and load classification head
        self._build_cls_head_if_needed()

        # Rebuild optimizer to include cls_head parameters
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
                
        # Ensure encoder remains frozen after loading
        self._freeze_encoder_parameters() 