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


class nnUNetTrainer_CLSHeadSepOpt(nnUNetTrainer):
    """
    2 separate optimizers for segmentation and classification
    Multi-task trainer that adds a case-level classification head using encoder features.
    The classification head consumes the encoder's bottleneck features via
    AdaptiveAvgPool3d -> Flatten -> Dropout -> Linear.

    Configuration via environment variables:
    - NNUNETV2_MT_NUM_CLS: number of classification classes (default: 2)
    - NNUNETV2_MT_LOSS_WEIGHT: lambda weight for classification loss (default: 0.3)
    - NNUNETV2_MT_MULTILABEL: '1' or 'true' for multilabel (BCEWithLogits), else multiclass (CrossEntropy)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mt_num_classes: int = int(os.environ.get('NNUNETV2_MT_NUM_CLS', '2'))
        self.mt_loss_weight: float = float(os.environ.get('NNUNETV2_MT_LOSS_WEIGHT', '0.3'))
        self.mt_multilabel: bool = os.environ.get('NNUNETV2_MT_MULTILABEL', '0').lower() in ('1', 'true', 't', 'yes')

        self.cls_head: nn.Module = None
        self.cls_loss_fn: nn.Module = None

    def build_cls_head(self):
        # this is hardcoded for now
        # Input channels for the head come from the encoder (320 channels based on our plan.json for 3d fullers)
        # change this based on the last feature size of the encoder in plans.json
        encoder_channels = 320
        
        print(f"Building classification head with {encoder_channels} input channels (encoder features)")
        
        # self.cls_head = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(encoder_channels, 256),
        #     # nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(256, self.mt_num_classes)
        # ).to(self.device) # for NB experiments

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

        cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  
            nn.Flatten(),
            nn.LayerNorm(encoder_channels),
            nn.Linear(encoder_channels, self.mt_num_classes)
        ).to(self.device) # simple LNorm experiment

        return cls_head

    def configure_optimizers_cls(self):
        # Ensure the classification head exists before creating its optimizer
        # self._build_cls_head_if_needed()
        # Optimizer and Scheduler for the main nnU-Net segmentation network
        # This uses SGD with PolyLRScheduler, as is standard for nnU-Net
        optimizer_seg = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )
        lr_scheduler_seg = PolyLRScheduler(optimizer_seg, self.initial_lr, self.num_epochs)

        # Optimizer and Scheduler for the classification head ---
        optimizer_cls = torch.optim.Adam(
            self.cls_head.parameters(),
            lr=self.initial_lr * 5 # Use a higher learning rate for the classification head
        )
        lr_scheduler_cls = CosineAnnealingLR(optimizer_cls, T_max=self.num_epochs)

        # Return both optimizers and schedulers as lists. This is a common pattern
        # for frameworks that support multi-optimizer setups.
        return optimizer_seg, optimizer_cls, lr_scheduler_seg, lr_scheduler_cls
    
    def initialize(self):
        super().initialize()
        self.cls_head = self.build_cls_head()
        weights_fn = torch.tensor([1.3548, 0.7925, 1.0], dtype=torch.float32).to(self.device)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=weights_fn)

        self.optimizer_seg, self.optimizer_cls, self.lr_scheduler_seg, self.lr_scheduler_cls = self.configure_optimizers_cls()


    def _compute_segmentation_loss_only(self, output, target):
        return self.loss(output, target)

    def _compute_classification_logits(self, encoder_features: torch.Tensor) -> torch.Tensor:
        # Use encoder features (320 channels) as input to the classification head
        # Ensure encoder features are float32 for the classification head
        if encoder_features.dtype != torch.float32:
            encoder_features = encoder_features.float()
        return self.cls_head(encoder_features)

    def train_step(self, batch: dict) -> dict:
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

        #Zero the gradients for BOTH optimizers
        self.optimizer_seg.zero_grad(set_to_none=True)
        self.optimizer_cls.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Forward pass - now returns seg_outputs AND encoder_features
            seg_outputs = self.network(data)
            with torch.no_grad():
                encoder_features = self.network.encoder(data)[-1]
            seg_loss = self._compute_segmentation_loss_only(seg_outputs, target)

            cls_loss = torch.tensor(0.0, device=self.device)
            cls_logits = self._compute_classification_logits(encoder_features)
            
            cls_target = cls_target.long()
            cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            # Calculate the total loss as a combined value
            total_loss = seg_loss + self.mt_loss_weight * cls_loss

        if self.grad_scaler is not None:
            # Perform backward pass on the total loss
            self.grad_scaler.scale(total_loss).backward()
            
            # Unscale gradients for BOTH optimizers
            self.grad_scaler.unscale_(self.optimizer_seg)
            self.grad_scaler.unscale_(self.optimizer_cls)
            
            # Grad norm clipping is applied to all parameters
            torch.nn.utils.clip_grad_norm_(list(self.network.parameters()) + list(self.cls_head.parameters()), 12)
            
            # Step BOTH optimizers
            self.grad_scaler.step(self.optimizer_seg)
            self.grad_scaler.step(self.optimizer_cls)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            
            # Grad norm clipping is applied to all parameters
            torch.nn.utils.clip_grad_norm_(list(self.network.parameters()) + list(self.cls_head.parameters()), 12)
            
            self.optimizer_seg.step()
            self.optimizer_cls.step()

        # Report total loss; optionally also return cls/seg components for debugging
        out = {'loss': total_loss.detach().cpu().numpy()}
        out['seg_loss'] = seg_loss.detach().cpu().numpy()
        out['cls_loss'] = cls_loss.detach().cpu().numpy()
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

            seg_loss = self._compute_segmentation_loss_only(seg_outputs, target)

            cls_loss = torch.tensor(0.0, device=self.device)
            if cls_target is not None and encoder_features is not None:
                cls_logits = self._compute_classification_logits(encoder_features)

                cls_target = cls_target.long()
                cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            total_loss = seg_loss + self.mt_loss_weight * cls_loss

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
            cls_logits = self._compute_classification_logits(encoder_features)
            # For single label multiclass, use argmax
            cls_pred = cls_logits.argmax(dim=1)
            cls_pred_np = cls_pred.detach().cpu().numpy()
            cls_target_np = cls_target.detach().cpu().numpy()
            f1_macro = f1_score(cls_target_np, cls_pred_np, average='macro', zero_division=0)
            
            print(f"F1 Macro: {f1_macro:.4f}")
        
        return out

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                # Get state dict from the original network (unwrapped)
                network_state = self.network.original_network.state_dict() if hasattr(self.network, 'original_network') else self.network.state_dict()
                
                checkpoint = {
                    'network_weights': (network_state if not self.is_ddp else self.network.module.original_network.state_dict()),
                    'optimizer_seg_state': self.optimizer_seg.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'cls_head_state': self.cls_head.state_dict() if self.cls_head is not None else None,
                    'optimizer_cls_state': self.optimizer_cls.state_dict(),
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
        if checkpoint.get('cls_head_state') is not None:
            self.cls_head.load_state_dict(checkpoint['cls_head_state'])

        # Rebuild optimizer to include cls_head parameters
        self.optimizer_seg, self.optimizer_cls, self.lr_scheduler_seg, self.lr_scheduler_cls = self.configure_optimizers_cls()
        self.optimizer_seg.load_state_dict(checkpoint['optimizer_seg_state'])

        if checkpoint.get('optimizer_cls_state') is not None:
            self.optimizer_cls.load_state_dict(checkpoint['optimizer_cls_state'])

        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
