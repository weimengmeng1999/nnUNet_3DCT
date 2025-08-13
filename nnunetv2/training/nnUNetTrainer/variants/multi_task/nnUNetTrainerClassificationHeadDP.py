import os
from typing import Tuple, Union, List

import numpy as np
import torch
from torch import nn
from torch import autocast
from torch import distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainer_CLSHeadDP(nnUNetTrainer):
    """
    Multi-task trainer that adds a case-level classification head on top of the segmentation network outputs.
    The classification head consumes the coarsest deep supervision segmentation output (last in the list) via
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
        self.mt_loss_weight: float = float(os.environ.get('NNUNETV2_MT_LOSS_WEIGHT', '0.05'))  # Lower default for small dataset
        # Force single label classification
        self.mt_multilabel: bool = False

        self.cls_head: nn.Module = None
        self.cls_loss_fn: nn.Module = None

    def _build_cls_head_if_needed(self):
        if self.cls_head is not None:
            return
        # Input channels for the head come from the segmentation head channels
        in_channels = self.label_manager.num_segmentation_heads
        
        # OPTIMIZED FOR SMALL DATASET (225 samples): Minimal parameters to prevent overfitting
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(p=0.7),  # High dropout for small dataset
            nn.Linear(in_channels, self.mt_num_classes),  # Direct mapping, no intermediate layers
        ).to(self.device)
        
        # Initialize with small weights to prevent saturation
        with torch.no_grad():
            for module in self.cls_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small initialization
                    nn.init.constant_(module.bias, 0)

        # Single label classification only - use CrossEntropyLoss with label smoothing
        self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def configure_optimizers(self):
        # Called in initialize() AND we call it again after attaching cls_head in on_train_start.
        # Ensure head exists before creating optimizer so its params are included.
        self._build_cls_head_if_needed()
        
        # OPTIMIZED FOR SMALL DATASET: Much lower learning rate and stronger L2 regularization
        seg_params = list(self.network.parameters())
        cls_params = list(self.cls_head.parameters())
        
        optimizer = torch.optim.SGD([
            {'params': seg_params, 'lr': self.initial_lr, 'weight_decay': self.weight_decay},
            {'params': cls_params, 'lr': self.initial_lr * 0.01, 'weight_decay': 0.01}  # Very low LR and high L2
        ], momentum=0.99, nesterov=True)
        
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def on_train_start(self):
        super().on_train_start()
        # Rebuild optimizer to include classification head
        self._build_cls_head_if_needed()
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

    def _compute_segmentation_loss_only(self, output, target):
        return self.loss(output, target)

    def _compute_classification_logits(self, seg_outputs: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        # Simplified approach for small datasets - no need for complex ensembling
        if isinstance(seg_outputs, (list, tuple)):
            # Use the first deep supervision output (highest resolution)
            feat = seg_outputs[0]
        else:
            feat = seg_outputs
        
        # Apply activation before feeding to classification head
        if self.label_manager.has_regions:
            feat = torch.sigmoid(feat)
        else:
            feat = torch.softmax(feat, dim=1)
        
        # FIXED: Ensure consistent dtype for mixed precision training
        # Convert to float32 to match classification head weights
        feat = feat.float()
            
        return self.cls_head(feat)

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

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_outputs = self.network(data)
            seg_loss = self._compute_segmentation_loss_only(seg_outputs, target)

            cls_loss = torch.tensor(0.0, device=self.device)
            if cls_target is not None:
                cls_logits = self._compute_classification_logits(seg_outputs)
                
                # FIXED: Better target handling and validation for SINGLE LABEL only
                cls_logits = self._compute_classification_logits(seg_outputs)
                
                # Single label classification - expect long targets of shape [B]
                if cls_target.ndim > 1:
                    if cls_target.shape[1] == 1:
                        cls_target = cls_target.squeeze(1)
                    else:
                        # If one-hot encoded, convert to class indices
                        cls_target = cls_target.argmax(dim=1)
                cls_target = cls_target.long()
                
                # FIXED: Validate target range
                if torch.any(cls_target >= self.mt_num_classes) or torch.any(cls_target < 0):
                    print(f"Warning: Invalid class labels found. Range: {cls_target.min()}-{cls_target.max()}, Expected: 0-{self.mt_num_classes-1}")
                    # Print actual values for debugging
                    print(f"Problematic targets: {cls_target.cpu().numpy()}")
                    cls_target = torch.clamp(cls_target, 0, self.mt_num_classes-1)
                
                cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            # OPTIMIZED FOR SMALL DATASET: Reduce classification loss weight to prevent overfitting
            # Start with very small weight and gradually increase
            epoch_factor = min(1.0, (self.current_epoch + 1) / 50)  # Ramp up over 50 epochs
            adaptive_weight = self.mt_loss_weight * 0.1 * epoch_factor  # Much smaller initial weight
            total_loss = seg_loss + adaptive_weight * cls_loss

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

        # FIXED: Add classification accuracy tracking
        out = {'loss': total_loss.detach().cpu().numpy()}
        out['seg_loss'] = seg_loss.detach().cpu().numpy()
        if cls_target is not None:
            out['cls_loss'] = cls_loss.detach().cpu().numpy()
            
            # Add accuracy tracking for SINGLE LABEL
            with torch.no_grad():
                cls_logits = self._compute_classification_logits(seg_outputs)
                cls_pred = cls_logits.argmax(dim=1)
                accuracy = (cls_pred == cls_target).float().mean().item()
                out['cls_accuracy'] = accuracy
                
                # Add prediction distribution for debugging
                pred_counts = torch.bincount(cls_pred, minlength=self.mt_num_classes)
                target_counts = torch.bincount(cls_target, minlength=self.mt_num_classes)
                out['pred_distribution'] = pred_counts.cpu().numpy()
                out['target_distribution'] = target_counts.cpu().numpy()
                
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
            seg_outputs = self.network(data)
            seg_loss = self._compute_segmentation_loss_only(seg_outputs, target)

            cls_loss = torch.tensor(0.0, device=self.device)
            if cls_target is not None:
                cls_logits = self._compute_classification_logits(seg_outputs)
                
                # Single label classification - expect long targets of shape [B] 
                if cls_target.ndim > 1:
                    if cls_target.shape[1] == 1:
                        cls_target = cls_target.squeeze(1)
                    else:
                        cls_target = cls_target.argmax(dim=1)
                cls_target = cls_target.long()
                cls_target = torch.clamp(cls_target, 0, self.mt_num_classes-1)
                cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            # FIXED: Add adaptive weight calculation for validation too
            epoch_factor = min(1.0, (self.current_epoch + 1) / 50)
            adaptive_weight = self.mt_loss_weight * 0.1 * epoch_factor
            total_loss = seg_loss + adaptive_weight * cls_loss

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
        if cls_target is not None:
            out['cls_loss'] = cls_loss.detach().cpu().numpy()
            
            # Add validation accuracy
            with torch.no_grad():
                if self.mt_multilabel:
                    cls_pred = (torch.sigmoid(cls_logits) > 0.5).float()
                    accuracy = ((cls_pred == cls_target).float().mean()).item()
                else:
                    cls_pred = cls_logits.argmax(dim=1)
                    accuracy = (cls_pred == cls_target).float().mean().item()
                out['cls_accuracy'] = accuracy
                
        return out

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                # Save base parts
                checkpoint = {
                    'network_weights': (self.network.module.state_dict() if self.is_ddp else self.network.state_dict()),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    # new: classification head
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
        
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
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
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if hasattr(self.network, 'module'):
                if hasattr(self.network.module, '_orig_mod'):  # OptimizedModule
                    self.network.module._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network.module.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        else:
            if hasattr(self.network, '_orig_mod'):  # OptimizedModule
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
                
        # Build and load classification head
        self._build_cls_head_if_needed()
        if checkpoint.get('cls_head_state') is not None:
            self.cls_head.load_state_dict(checkpoint['cls_head_state'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])