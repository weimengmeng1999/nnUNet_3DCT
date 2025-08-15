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
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import shutil
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.collate_outputs import collate_outputs

from time import time, sleep

class nnUNetTrainer_CLSHead(nnUNetTrainer):
    """
    Multi-task trainer that adds a case-level classification head using encoder features.
    The classification head consumes the encoder's bottleneck features via
    AdaptiveAvgPool3d -> Flatten -> Dropout -> Linear.

    Configuration via environment variables:
    - NNUNETV2_MT_NUM_CLS: number of classification classes (default: 2)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mt_num_classes: int = int(os.environ.get('NNUNETV2_MT_NUM_CLS', '2'))

        self.cls_head: nn.Module = None
        self.cls_loss_fn: nn.Module = None

    def _build_cls_head_if_needed(self):
        
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

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  
            nn.Flatten(),
            # nn.LayerNorm(encoder_channels),
            nn.Linear(encoder_channels, self.mt_num_classes)
        ).to(self.device) # try classification head + encoder only experiment

        self.cls_loss_fn = nn.CrossEntropyLoss()

    # def configure_optimizers(self):
    #     # Called in initialize() AND we call it again after attaching cls_head in on_train_start.
    #     # Ensure head exists before creating optimizer so its params are included.
    #     self._build_cls_head_if_needed()
    #     # optimizer = torch.optim.SGD(
    #     #     list(self.network.parameters()) + list(self.cls_head.parameters()),
    #     #     self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
    #     # )
    #     optimizer = torch.optim.SGD([
    #                                 {'params': self.network.parameters()},
    #                                 {'params': self.cls_head.parameters(), 'lr': self.initial_lr * 5}
    #                             ], lr=self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
    #     from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
    #     lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
    #     return optimizer, lr_scheduler

    def configure_optimizers_cls_head(self):
        # Called in initialize() AND we call it again after attaching cls_head in on_train_start.
        # Ensure head exists before creating optimizer so its params are included.
        # optimizer = torch.optim.SGD(
        #     list(self.network.encoder.parameters()) + list(self.cls_head.parameters()),
        #     1e-4, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
        # )
        params = list(self.network.encoder.parameters()) + list(self.cls_head.parameters())
        # print(f"Params: {list(self.network.encoder.parameters())}")
        print(f"Params size encoder: {len(list(self.network.encoder.parameters()))}")
        print(f"Params size cls_head: {len(list(self.cls_head.parameters()))}")
        print(f"Params size total: {len(params)}")
        optimizer = torch.optim.Adam(
            params,
            lr=1e-8 # Use a higher learning rate for the classification head
        )
        return optimizer


    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_train_start(self):
        # super().on_train_start()
        if not self.was_initialized:
            self.initialize()

        # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
        # which may not be present  when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.local_rank == 0:
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder,
                overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
                verify=True)

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

        # Rebuild optimizer to include classification head
        self._build_cls_head_if_needed()
        self.optimizer = self.configure_optimizers_cls_head()

    def _compute_classification_logits(self, encoder_features: torch.Tensor) -> torch.Tensor:
        # Use encoder features (320 channels) as input to the classification head
        # Ensure encoder features are float32 for the classification head
        if encoder_features.dtype != torch.float32:
            encoder_features = encoder_features.float()
        return self.cls_head(encoder_features)

    def train_step(self, batch: dict) -> dict:

        data = batch['data']
        cls_target = batch['cls_target']

        data = data.to(self.device, non_blocking=True)
        cls_target = cls_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        encoder_features = self.network.encoder(data)[-1]
        cls_logits = self._compute_classification_logits(encoder_features)
        
        cls_target = cls_target.long()
        total_loss = self.cls_loss_fn(cls_logits, cls_target)

        total_loss.backward()
        self.optimizer.step()

        # Report total loss; optionally also return cls/seg components for debugging
        out = {'loss': total_loss.detach().cpu().numpy()}

        return out

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        cls_target = batch['cls_target']

        data = data.to(self.device, non_blocking=True)
        cls_target = cls_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        encoder_features = self.network.encoder(data)[-1]     
        cls_logits = self._compute_classification_logits(encoder_features)
        
        cls_target = cls_target.long()
        total_loss = self.cls_loss_fn(cls_logits, cls_target)

        out = {'loss': total_loss.detach().cpu().numpy()}

        
        # Calculate F1 score for classification if we have classification targets
        # if cls_target is not None and encoder_features is not None:
        #     cls_logits = self._compute_classification_logits(encoder_features)
        #     # For single label multiclass, use argmax
        #     cls_pred = cls_logits.argmax(dim=1)
        #     # Convert to numpy for sklearn
        #     cls_pred_np = cls_pred.detach().cpu().numpy()
        #     cls_target_np = cls_target.detach().cpu().numpy()
        #     # Calculate macro F1 score
        #     f1_macro = f1_score(cls_target_np, cls_pred_np, average='macro', zero_division=0)
            
        #     print(f"F1 Macro: {f1_macro:.4f}")
        
        return out

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)


        loss_here = np.mean(outputs_collated['loss'])
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

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
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        assert False, 'load_checkpoint is not supported for this trainer'
        print('load_checkpoint')
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