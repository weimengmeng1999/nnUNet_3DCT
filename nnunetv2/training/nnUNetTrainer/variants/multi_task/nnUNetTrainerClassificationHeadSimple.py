
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
from sklearn.metrics import accuracy_score, f1_score
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p


from nnunetv2.utilities.collate_outputs import collate_outputs

from time import time


class nnUNetTrainer_CLSHeadSimple(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mt_num_classes = int(os.environ.get('NNUNETV2_MT_NUM_CLS', '3'))
        self.mt_loss_weight = float(os.environ.get('NNUNETV2_MT_LOSS_WEIGHT', '0.3'))
        # self.pre_checkpoint_path = os.environ.get('NNUNETV2_PRE_CHECKPOINT_PATH', '/home/mengwei/Downloads/nnUNet_code/pretrain/checkpoint_best.pth')
        
        # Calculate proper class weights based on your training data
        self.cls_head = None
        self.cls_loss_fn = None
        # self.gradient_accumulation_steps = 4  # Effective batch_size=8

    def build_cls_head(self):
        # cls_head = Classifier3D(self.mt_num_classes).to(self.device)
        # cls_head = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),  
        #     nn.Flatten(),
        #     nn.Linear(320, self.mt_num_classes)
        # )
        #simple ver2
        cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  
            nn.Flatten(),
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, self.mt_num_classes)
        )
        return cls_head
    
    def configure_optimizers_cls(self):
        optimizer = torch.optim.SGD(
            list(self.network.encoder.parameters()) + list(self.cls_head.parameters()),
            1e-3, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
        )
        
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def initialize(self):
        super().initialize()
        self.cls_head = self.build_cls_head()
        self.cls_head.to(self.device)
        weights_fn = torch.tensor([1.3548, 0.7925, 1.0], dtype=torch.float32).to(self.device)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=weights_fn)

        self.optimizer, self.lr_scheduler = self.configure_optimizers_cls()
        

        # add the following for stage 1 pre-trained weights
        # checkpoint = torch.load(self.pre_checkpoint_path, map_location=self.device, weights_only=False)

        # new_state_dict = {}
        # for k, value in checkpoint['network_weights'].items():
        #     key = k
        #     if key not in self.network.state_dict().keys() and key.startswith('module.'):
        #         key = key[7:]
        #     new_state_dict[key] = value
            
        # # # Load the weights
        # self.network._orig_mod.load_state_dict(new_state_dict)

        # Mark as loaded to avoid reloading every step
        # self._pretrained_loaded = True
        # print("Pretrained weights loaded successfully!")
    
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
            encoder_features = self.network.encoder(data)[-1]
            cls_loss = torch.tensor(0.0, device=self.device)
            cls_logits = self.cls_head(encoder_features)
            cls_loss = self.cls_loss_fn(cls_logits, cls_target)
            total_loss = cls_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.network.encoder.parameters()) + list(self.cls_head.parameters()), 12)
            # torch.nn.utils.clip_grad_norm_(list(self.network.parameters()), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.network.encoder.parameters()) + list(self.cls_head.parameters()), 12)
            # torch.nn.utils.clip_grad_norm_(list(self.network.parameters()), 12)

            self.optimizer.step()

        # Metrics
        out = {
            'loss': total_loss.detach().cpu().numpy(),
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
            encoder_features = self.network.encoder(data)[-1]        

            cls_loss = torch.tensor(0.0, device=self.device)

            cls_logits = self.cls_head(encoder_features)
            cls_loss = self.cls_loss_fn(cls_logits, cls_target)
            total_loss = cls_loss

            pred_class = cls_logits.argmax(dim=1).detach().cpu().numpy()
            true_class = cls_target.detach().cpu().numpy()

        out = {'loss': total_loss.detach().cpu().numpy(),
                'true_class': true_class,       # for macro-F1
                'pred_class': pred_class
                }

        return out

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        all_true = np.asarray(outputs_collated['true_class']).reshape(-1)  # ensure shape (N,)
        all_pred = np.asarray(outputs_collated['pred_class']).reshape(-1)  # ensure shape (N,)

        acc = accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred, average='macro')

        self.logger.log('val_classification_acc', acc, self.current_epoch)
        self.logger.log('val_classification_f1', f1, self.current_epoch)


        loss_here = np.mean(outputs_collated['loss'])
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('val_classification_acc', np.round(self.logger.my_fantastic_logging['val_classification_acc'][-1], decimals=4))
        self.print_to_log_file('val_classification_f1', np.round(self.logger.my_fantastic_logging['val_classification_f1'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if not hasattr(self, "_best_cls_f1"):
            self._best_cls_f1 = None

        # handle best checkpointing for classification
        current_f1 = self.logger.my_fantastic_logging['val_classification_f1'][-1]
        # if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
        if self._best_cls_f1 is None or current_f1 > self._best_cls_f1:
            self._best_cls_f1 = current_f1
            filename = f'checkpoint_cls_best.pth'
            self.save_checkpoint(join(self.output_folder, filename))
            self.print_to_log_file(f"new best macro F1: {np.round(current_f1, 4)}")

        if self.local_rank == 0:
            print("plotting progress png")
            self.logger.plot_progress_png_ocls(self.output_folder)

        self.current_epoch += 1

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
        self.build_cls_head_if_needed()
        if checkpoint.get('cls_head_state') is not None:
            self.cls_head.load_state_dict(checkpoint['cls_head_state'])

        # Rebuild optimizer to include cls_head parameters
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])