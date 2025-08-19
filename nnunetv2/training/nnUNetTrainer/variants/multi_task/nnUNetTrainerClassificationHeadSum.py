
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

from torch.nn import init
from time import time

#ver1
# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features"""
#     def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32]):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool3d(1)
#         self.fc_layers = nn.ModuleList([
#             nn.Linear(ch, num_classes) for ch in encoder_channels
#         ])

#     def forward(self, *features):
#         outputs = []
#         for feat, fc in zip(features, self.fc_layers):
#             pooled = self.avgpool(feat).flatten(1)
#             outputs.append(fc(pooled))
#         return sum(outputs)  # sum logits from all scales

#ver2
#alternative version of classifier head with max pooling and linear layer for concatenation of all scales features
# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features"""
#     def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32]):
#         super().__init__()
#         self.pool = nn.AdaptiveMaxPool3d(1)  # global max pooling
#         self.fc_layers = nn.ModuleList([
#             nn.Linear(ch, num_classes) for ch in encoder_channels
#         ])
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features):
#         logits_list = []
#         for feat, fc in zip(features, self.fc_layers):
#             pooled = self.pool(feat).flatten(1)
#             logits_list.append(fc(pooled))  # per-scale logits

#         concat_logits = torch.cat(logits_list, dim=1)
#         return self.final_fc(concat_logits)

#ver3
#alternative version of classifier head with max pooling and avg pooling and linear layer for concatenation of all scales features
# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features with GAP + GMP pooling"""
#     def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32]):
#         super().__init__()
#         # input dimension doubles because we concat GAP + GMP
#         self.fc_layers = nn.ModuleList([
#             nn.Linear(ch * 2, num_classes) for ch in encoder_channels
#         ])
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features):
#         logits_list = []
#         for feat, fc in zip(features, self.fc_layers):
#             avg_pooled = F.adaptive_avg_pool3d(feat, 1).flatten(1)
#             max_pooled = F.adaptive_max_pool3d(feat, 1).flatten(1)
#             pooled = torch.cat([avg_pooled, max_pooled], dim=1)
#             logits_list.append(fc(pooled))  # per-scale logits

#         concat_logits = torch.cat(logits_list, dim=1)
#         return self.final_fc(concat_logits)

#ver4
#ver 3 with dropout
# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features with GAP + GMP pooling"""
#     def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32]):
#         super().__init__()
#         # input dimension doubles because we concat GAP + GMP
#         self.fc_layers = nn.ModuleList([
#             nn.Linear(ch * 2, num_classes) for ch in encoder_channels
#         ])
#         self.dropout = nn.Dropout(p=0.3)
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features):
#         logits_list = []
#         for feat, fc in zip(features, self.fc_layers):
#             avg_pooled = F.adaptive_avg_pool3d(feat, 1).flatten(1)
#             max_pooled = F.adaptive_max_pool3d(feat, 1).flatten(1)
#             pooled = torch.cat([avg_pooled, max_pooled], dim=1)
#             logits_list.append(fc(pooled))  # per-scale logits

#         concat_logits = torch.cat(logits_list, dim=1)
#         concat_logits = self.dropout(concat_logits)
#         return self.final_fc(concat_logits)

#ver5 
# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features with GMP pooling + per-scale MLP"""
#     def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32], hidden_dim=128):
#         super().__init__()
#         # Per-scale MLPs: input = ch (GMP only), hidden = hidden_dim, output = num_classes
#         self.mlp_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(ch, ch // 2),
#                 nn.ReLU(),
#                 nn.Linear(ch // 2, num_classes)
#             ) for ch in encoder_channels
#         ])
#         # Final FC combines all per-scale logits
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features):
#         logits_list = []
#         for feat, mlp in zip(features, self.mlp_layers):
#             max_pooled = F.adaptive_max_pool3d(feat, 1).flatten(1)  # GMP only
#             logits_list.append(mlp(max_pooled))  # per-scale logits via MLP

#         concat_logits = torch.cat(logits_list, dim=1)
#         return self.final_fc(concat_logits)

#ver6
# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features"""
#     def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32]):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool3d(1)
#         # self.fc_layers = nn.ModuleList([
#         #     nn.Linear(ch, num_classes) for ch in encoder_channels
#         # ])
#         self.fc_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(ch, ch // 2),
#                 nn.ReLU(),
#                 nn.Linear(ch // 2, num_classes)
#             ) for ch in encoder_channels
#         ])
    # def forward(self, *features):
    #     outputs = []
    #     for feat, fc in zip(features, self.fc_layers):
    #         pooled = self.avgpool(feat).flatten(1)
    #         outputs.append(fc(pooled))
    #     return sum(outputs)  # sum logits from all scales

#ver7
class Classifier3D(nn.Module):
    """3D Classifier for multi-scale nnUNet features with Mask-guided ROI pooling + per-scale MLP"""
    def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32], hidden_dim=128, grid_size=(4,4,4)):
        super().__init__()
        self.grid_size = grid_size  # adaptive pooling grid
        # Per-scale MLPs: input = ch * grid_volume, hidden = ch//2, output = num_classes
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch * grid_size[0] * grid_size[1] * grid_size[2], ch // 2),
                nn.ReLU(),
                nn.Linear(ch // 2, num_classes)
            ) for ch in encoder_channels
        ])
        # Final FC combines all per-scale logits
        self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

    def forward(self, *features):
        """
        features: list of encoder features at multiple scales (B x C x D x H x W)
        masks: segmentation masks (B x 1 x D x H x W), optional
        """
        logits_list = []
        for feat, mlp in zip(features, self.mlp_layers):
            # Adaptive ROI pooling to small 3D grid
            pooled = F.adaptive_max_pool3d(feat, output_size=self.grid_size)
            pooled = pooled.view(feat.shape[0], -1)  # flatten to B x (C*grid_volume)

            logits_list.append(mlp(pooled))  # per-scale logits

        concat_logits = torch.cat(logits_list, dim=1)
        return self.final_fc(concat_logits)

#ver8
# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features with small 3D pooled grid per scale"""
#     def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32], grid_size=(4, 4, 4)):
#         super().__init__()
#         self.grid_size = grid_size  # adaptive pooling grid
        
#         # Per-scale MLPs: input = ch * grid_volume, hidden = ch//2, output = num_classes
#         self.fc_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(ch * grid_size[0] * grid_size[1] * grid_size[2], ch // 2),
#                 nn.ReLU(),
#                 nn.Linear(ch // 2, num_classes)
#             ) for ch in encoder_channels
#         ])

#     def forward(self, *features):
#         logits_list = []
#         for feat, fc in zip(features, self.fc_layers):
#             # Adaptive 3D pooling to small grid
#             pooled = F.adaptive_max_pool3d(feat, output_size=self.grid_size)
#             pooled = pooled.view(feat.shape[0], -1)  # flatten: B x (C*grid_volume)
            
#             logits_list.append(fc(pooled))  # per-scale logits

#         return sum(logits_list)  # sum logits from all scales

#ver9
# class Classifier3D(nn.Module):
#     """
#     3D Classifier for multi-scale nnUNet features
#     Mask-guided ROI pooling + grid pooling + per-scale self-attention + MLP
#     """
#     def __init__(
#         self,
#         num_classes,
#         encoder_channels=[320, 320, 256, 128, 64, 32],
#         hidden_dim=128,
#         grid_size=(4, 4, 4),
#         attn_heads=4,
#         attn_dim=128
#     ):
#         super().__init__()
#         self.grid_size = grid_size
#         self.num_classes = num_classes

#         self.proj_layers = nn.ModuleList()
#         self.attn_layers = nn.ModuleList()
#         self.mlp_layers = nn.ModuleList()

#         for ch in encoder_channels:
#             # Input dimension after flattening pooled grid
#             pooled_dim = ch * grid_size[0] * grid_size[1] * grid_size[2]

#             # Linear projection to attention embedding
#             self.proj_layers.append(nn.Linear(ch, attn_dim))

#             # Multi-head attention
#             self.attn_layers.append(nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True))

#             # Per-scale MLP
#             self.mlp_layers.append(
#                 nn.Sequential(
#                     nn.Linear(attn_dim, hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(hidden_dim, num_classes)
#                 )
#             )

#         # Final FC to combine per-scale logits
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features):
#         """
#         features: list of encoder features at multiple scales (B x C x D x H x W)
#         masks: optional segmentation masks (B x 1 x D x H x W)
#         """
#         logits_list = []

#         for i, (feat, proj, attn, mlp) in enumerate(zip(features, self.proj_layers, self.attn_layers, self.mlp_layers)):
#             B, C, D, H, W = feat.shape

#             # Adaptive grid pooling
#             pooled = F.adaptive_max_pool3d(feat, output_size=self.grid_size)  # B x C x d x h x w
#             B, C, d, h, w = pooled.shape

#             # Flatten spatial positions: B x L x C
#             pooled_flat = pooled.view(B, C, -1).transpose(1, 2)  # B x L x C

#             # Linear projection per position
#             pooled_attn = proj(pooled_flat)  # B x L x attn_dim

#             # Self-attention over spatial positions
#             attn_out, _ = attn(pooled_attn, pooled_attn, pooled_attn)  # B x L x attn_dim
#             attn_out = attn_out.mean(dim=1)  # B x attn_dim

#             # Per-scale MLP
#             logits_list.append(mlp(attn_out))  # B x num_classes

#         # Concatenate per-scale logits and final FC
#         concat_logits = torch.cat(logits_list, dim=1)
#         return self.final_fc(concat_logits)

#ver10 
# class Classifier3D(nn.Module):
#     """
#     3D Classifier for multi-scale nnUNet features
#     Mask-guided ROI pooling + grid pooling + per-scale self-attention + MLP
#     """
#     def __init__(
#         self,
#         num_classes,
#         encoder_channels=[320, 320, 256, 128, 64, 32],
#         hidden_dim=128,
#         grid_size=(4, 4, 4),
#         attn_heads=4,
#         attn_dim=128
#     ):
#         super().__init__()
#         self.grid_size = grid_size
#         self.num_classes = num_classes

#         self.proj_layers = nn.ModuleList()
#         self.attn_layers = nn.ModuleList()
#         self.mlp_layers = nn.ModuleList()

#         for ch in encoder_channels:
#             # Input dimension after flattening pooled grid
#             pooled_dim = ch * grid_size[0] * grid_size[1] * grid_size[2]

#             # Linear projection to attention embedding
#             self.proj_layers.append(nn.Linear(ch, attn_dim))

#             # Multi-head attention
#             self.attn_layers.append(nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True))

#             # Per-scale MLP
#             self.mlp_layers.append(
#                 nn.Sequential(
#                     nn.Linear(attn_dim, hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(hidden_dim, num_classes)
#                 )
#             )

#         # Final FC to combine per-scale logits
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features, masks=None):
#         """
#         features: list of encoder features at multiple scales (B x C x D x H x W)
#         masks: optional segmentation masks (B x 1 x D x H x W)
#         """
#         logits_list = []

#         for i, (feat, proj, attn, mlp) in enumerate(zip(features, self.proj_layers, self.attn_layers, self.mlp_layers)):
#             B, C, D, H, W = feat.shape

#             # Apply mask if provided
#             if masks is not None:
#                 mask_resized = F.interpolate(masks.float(), size=(D, H, W), mode='trilinear', align_corners=False)
#                 feat = feat * mask_resized

#             # Adaptive grid pooling
#             pooled = F.adaptive_max_pool3d(feat, output_size=self.grid_size)  # B x C x d x h x w
#             B, C, d, h, w = pooled.shape

#             # Flatten spatial positions: B x L x C
#             pooled_flat = pooled.view(B, C, -1).transpose(1, 2)  # B x L x C

#             # Linear projection per position
#             pooled_attn = proj(pooled_flat)  # B x L x attn_dim

#             # Self-attention over spatial positions
#             attn_out, _ = attn(pooled_attn, pooled_attn, pooled_attn)  # B x L x attn_dim
#             attn_out = attn_out.mean(dim=1)  # B x attn_dim

#             # Per-scale MLP
#             logits_list.append(mlp(attn_out))  # B x num_classes

#         # Concatenate per-scale logits and final FC
#         concat_logits = torch.cat(logits_list, dim=1)
#         return self.final_fc(concat_logits)

#ver11
# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features with Mask-guided ROI pooling + per-scale MLP"""
#     def __init__(self, num_classes, encoder_channels=[320, 320, 256, 128, 64, 32], hidden_dim=128, grid_size=(4,4,4)):
#         super().__init__()
#         self.grid_size = grid_size  # adaptive pooling grid
#         # Per-scale MLPs: input = ch * grid_volume, hidden = ch//2, output = num_classes
#         self.mlp_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(ch * grid_size[0] * grid_size[1] * grid_size[2], ch // 2),
#                 nn.ReLU(),
#                 nn.Linear(ch // 2, num_classes)
#             ) for ch in encoder_channels
#         ])
#         # Final FC combines all per-scale logits
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features, masks=None):
#         """
#         features: list of encoder features at multiple scales (B x C x D x H x W)
#         masks: segmentation masks (B x 1 x D x H x W), optional
#         """
#         logits_list = []
#         for feat, mlp in zip(features, self.mlp_layers):
#             if masks is not None:
#                 # Resize mask to feature spatial size
#                 mask_resized = F.interpolate(masks.float(), size=feat.shape[2:], mode='trilinear', align_corners=False)
#                 feat = feat * mask_resized  # apply mask

#             # Adaptive ROI pooling to small 3D grid
#             pooled = F.adaptive_max_pool3d(feat, output_size=self.grid_size)
#             pooled = pooled.view(feat.shape[0], -1)  # flatten to B x (C*grid_volume)

#             logits_list.append(mlp(pooled))  # per-scale logits

#         concat_logits = torch.cat(logits_list, dim=1)
#         return self.final_fc(concat_logits)

#ver13: VER7 with mask pooling

# class Classifier3D(nn.Module):
#     """3D Classifier for multi-scale nnUNet features with lesion-only mask-guided pooling + per-scale MLP"""
#     def __init__(self, num_classes, encoder_channels=[320, 256, 128, 64, 32], grid_size=(4,4,4)):
#         super().__init__()
#         self.grid_size = grid_size
#         self.mlp_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(ch * grid_size[0] * grid_size[1] * grid_size[2], ch // 2),
#                 nn.ReLU(),
#                 nn.Linear(ch // 2, num_classes)
#             ) for ch in encoder_channels
#         ])
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features, masks=None):
#         """
#         features: list of encoder features at multiple scales (B x C x D x H x W)
#         masks: list of deep supervision masks, same number of scales as features
#                each mask: (B x 1 x D x H x W)
#         """
#         logits_list = []
#         for i, (feat, mlp) in enumerate(zip(features, self.mlp_layers)):
#             if masks is not None:
#                 # use corresponding scale mask
#                 mask_scale = masks[-(i+1)].float()  # B x 1 x D x H x W
#                 mask_lesion = (mask_scale > 0).float()
#                 feat = feat * mask_lesion

#             # Adaptive pooling to grid
#             pooled = F.adaptive_max_pool3d(feat, output_size=self.grid_size)
#             pooled = pooled.view(feat.shape[0], -1)
#             logits_list.append(mlp(pooled))

#         concat_logits = torch.cat(logits_list, dim=1)
#         return self.final_fc(concat_logits)

#ver12 without mask pooling
# class Classifier3D(nn.Module):
#     """
#     3D Classifier for multi-scale nnUNet features
#     Mask-guided ROI pooling + grid pooling + per-scale self-attention + MLP
#     """
#     def __init__(
#         self,
#         num_classes,
#         encoder_channels=[320, 256, 128, 64, 32],
#         hidden_dim=128,
#         grid_size=(4, 4, 4),
#         attn_heads=4,
#         attn_dim=128
#     ):
#         super().__init__()
#         self.grid_size = grid_size
#         self.num_classes = num_classes

#         self.proj_layers = nn.ModuleList()
#         self.attn_layers = nn.ModuleList()
#         self.mlp_layers = nn.ModuleList()

#         for ch in encoder_channels:
#             # Input dimension after flattening pooled grid
#             pooled_dim = ch * grid_size[0] * grid_size[1] * grid_size[2]

#             # Linear projection to attention embedding
#             self.proj_layers.append(nn.Linear(ch, attn_dim))

#             # Multi-head attention
#             self.attn_layers.append(nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True))

#             # Per-scale MLP
#             self.mlp_layers.append(
#                 nn.Sequential(
#                     nn.Linear(attn_dim, hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(hidden_dim, num_classes)
#                 )
#             )

#         # Final FC to combine per-scale logits
#         self.final_fc = nn.Linear(num_classes * len(encoder_channels), num_classes)

#     def forward(self, *features, masks=None):
#         """
#         features: list of encoder features at multiple scales (B x C x D x H x W)
#         masks: optional segmentation masks (B x 1 x D x H x W)
#         """
#         logits_list = []

#         for i, (feat, proj, attn, mlp) in enumerate(zip(features, self.proj_layers, self.attn_layers, self.mlp_layers)):
#             B, C, D, H, W = feat.shape

#             # Apply mask if provided
#             if masks is not None:
#                 # use corresponding scale mask
#                 mask_scale = masks[-(i+1)].float()  # B x 1 x D x H x W
#                 mask_lesion = (mask_scale > 0).float()
#                 feat = feat * mask_lesion

#             # Adaptive grid pooling
#             pooled = F.adaptive_max_pool3d(feat, output_size=self.grid_size)  # B x C x d x h x w
#             B, C, d, h, w = pooled.shape

#             # Flatten spatial positions: B x L x C
#             pooled_flat = pooled.view(B, C, -1).transpose(1, 2)  # B x L x C

#             # Linear projection per position
#             pooled_attn = proj(pooled_flat)  # B x L x attn_dim

#             # Self-attention over spatial positions
#             attn_out, _ = attn(pooled_attn, pooled_attn, pooled_attn)  # B x L x attn_dim
#             attn_out = attn_out.mean(dim=1)  # B x attn_dim

#             # Per-scale MLP
#             logits_list.append(mlp(attn_out))  # B x num_classes

#         # Concatenate per-scale logits and final FC
#         concat_logits = torch.cat(logits_list, dim=1)
#         return self.final_fc(concat_logits)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 

class nnUNetTrainer_CLSHeadSum(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mt_num_classes = int(os.environ.get('NNUNETV2_MT_NUM_CLS', '3'))
        self.mt_loss_weight = float(os.environ.get('NNUNETV2_MT_LOSS_WEIGHT', '0.3'))
        self.pre_checkpoint_path = os.environ.get('NNUNETV2_PRE_CHECKPOINT_PATH', 
        '/nfs/home/mwei/nnUNet_data/nnUNet_results/Dataset001_3DCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth')
        
        # Calculate proper class weights based on your training data
        self.cls_head = None
        self.cls_loss_fn = None
        # self.gradient_accumulation_steps = 4  # Effective batch_size=8

    def build_cls_head(self):
        cls_head = Classifier3D(self.mt_num_classes).to(self.device)

        return cls_head
    
    def configure_optimizers_cls(self):
        #train from stage 1 checkpoint
        optimizer = torch.optim.SGD([
            {'params': self.network.parameters(), 'lr': self.initial_lr},
            {'params': self.cls_head.parameters(), 'lr': self.initial_lr * 10}  # Higher LR for head
        ], weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        #train from stage 1 checkpoint with adamw
        # optimizer = torch.optim.AdamW([
        #     {'params': self.network.parameters(), 'lr': self.initial_lr, 'weight_decay': self.weight_decay},
        #     {'params': self.cls_head.parameters(), 'lr': self.initial_lr * 10, 'weight_decay': self.weight_decay}
        # ])
        #train from scratch
#         optimizer = torch.optim.SGD(
#                 list(self.network.parameters()) + list(self.cls_head.parameters()),
#                 lr=self.initial_lr,
#                 weight_decay=self.weight_decay,
#                 momentum=0.99,
#                 nesterov=True
# )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def initialize(self):
        super().initialize()
        self.cls_head = self.build_cls_head()
        # weights_fn = torch.tensor([1.3548, 0.7925, 1.0], dtype=torch.float32).to(self.device)
        # self.cls_loss_fn = nn.CrossEntropyLoss(weight=weights_fn)
        # self.cls_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        self.cls_loss_fn = nn.CrossEntropyLoss()

        self.initial_lr = 1e-3
        self.num_epochs = 200
        self.mt_loss_weight = 0.3 #0.5 for ver12 experiment

        self.optimizer, self.lr_scheduler = self.configure_optimizers_cls()
        #add for using checkpoint for stage 1 trained only for segmentation
        checkpoint = torch.load(self.pre_checkpoint_path, map_location=self.device, weights_only=False)

        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
            
        # # Load the weights
        self.network._orig_mod.load_state_dict(new_state_dict)

        # Mark as loaded to avoid reloading every step
        self._pretrained_loaded = True
        print("Pretrained weights loaded successfully!")
    
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
            encoder_features = self.network.encoder(data)  # list of encoder stage outputs

            # Get bottleneck encoder feature
            decoder_features, seg_outputs = [], []
            bottleneck_feature = encoder_features[-1]
            y = bottleneck_feature
            for i, stage in enumerate(self.network.decoder.stages):

                y = self.network.decoder.transpconvs[i](y)
                y = torch.cat((y, encoder_features[-(i+2)]), 1) #NOTE??
                y = stage(y)
                decoder_features.append(y)

                if self.enable_deep_supervision:
                    seg_outputs.append(self.network.decoder.seg_layers[i](y))
                elif i == (len(self.network.decoder.stages) - 1):
                    seg_outputs.append(self.network.decoder.seg_layers[-1](y))


            # invert seg outputs so that the largest segmentation prediction is returned first
            seg_outputs = seg_outputs[::-1]
            if not self.enable_deep_supervision:
                seg_outputs = seg_outputs[0]

            seg_loss = self._compute_segmentation_loss_only(seg_outputs, target)
            
            cls_loss = torch.tensor(0.0, device=self.device)
            cls_logits = self.cls_head(*decoder_features)
            #with mask guided pooling
            # cls_logits = self.cls_head(*decoder_features, masks=target)
            cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            total_loss = seg_loss + self.mt_loss_weight * cls_loss
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
            # Single forward pass through encoder
            encoder_features = self.network.encoder(data)  # list of encoder stage outputs

            # Get bottleneck encoder feature
            decoder_features, seg_outputs = [], []
            bottleneck_feature = encoder_features[-1]
            y = bottleneck_feature
            for i, stage in enumerate(self.network.decoder.stages):

                y = self.network.decoder.transpconvs[i](y)
                y = torch.cat((y, encoder_features[-(i+2)]), 1) #NOTE??
                y = stage(y)
                decoder_features.append(y)

                if self.enable_deep_supervision:
                    seg_outputs.append(self.network.decoder.seg_layers[i](y))
                elif i == (len(self.network.decoder.stages) - 1):
                    seg_outputs.append(self.network.decoder.seg_layers[-1](y))


            # invert seg outputs so that the largest segmentation prediction is returned first
            seg_outputs = seg_outputs[::-1]
            if not self.enable_deep_supervision:
                seg_outputs = seg_outputs[0]


            seg_loss = self._compute_segmentation_loss_only(seg_outputs, target)

            cls_loss = torch.tensor(0.0, device=self.device)
            cls_logits = self.cls_head(*decoder_features)
            # with mask guided pooling
            # cls_logits = self.cls_head(*decoder_features, masks=target)
            cls_loss = self.cls_loss_fn(cls_logits, cls_target)

            # total_loss = seg_loss + self.mt_loss_weight * cls_loss
            total_loss = seg_loss + self.mt_loss_weight * cls_loss
            # total_loss = seg_loss + cls_loss

            pred_class = cls_logits.argmax(dim=1).detach().cpu().numpy()
            true_class = cls_target.detach().cpu().numpy()

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
        out['true_class'] = true_class
        out['pred_class'] = pred_class

        return out

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

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
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file('val_classification_acc', np.round(self.logger.my_fantastic_logging['val_classification_acc'][-1], decimals=4))
        self.print_to_log_file('val_classification_f1', np.round(self.logger.my_fantastic_logging['val_classification_f1'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
        
        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

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
            self.logger.plot_progress_png_wcls(self.output_folder)

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