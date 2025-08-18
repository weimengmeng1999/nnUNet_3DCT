#check data sanity
# export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
# export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
# export nnUNet_results="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results"
# nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
# Run training with correct nnU-Net v2 syntax
# Format: nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options]

#train segmentation and classification together
# echo "Starting nnU-Net v2 training..."
# export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
# export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
# export nnUNet_results="/nfs/home/mwei/nnUNet_results/nnUNet_results_oclsplusencoder_adam"
# export NNUNETV2_MT_NUM_CLS=3 #your number of classes
# export NNUNETV2_MT_LOSS_WEIGHT=1 #lambda for cls loss 
# nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHead -p nnUNetPlans --npz

#train segmentation and classification together with dropout design
# echo "Starting nnU-Net v2 training..."
# export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
# export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
# export nnUNet_results="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wclsDP"
# export NNUNETV2_MT_NUM_CLS=3 #your number of classes
# export NNUNETV2_MT_LOSS_WEIGHT=0.05 #lambda for cls loss 
# nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHeadDP -p nnUNetPlans --npz

# only train classification head- freeze pre-trained segmentation encoder
# echo "Starting nnU-Net v2 training..."
# export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
# export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
# export nnUNet_results="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_ocls_simple_LNorm"
# export NNUNETV2_MT_NUM_CLS=3 #your number of classes
# export NNUNETV2_MT_LOSS_WEIGHT=1 #lambda for cls loss 
# export NNUNETV2_PRE_CHECKPOINT_PATH="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wcls_dp07/Dataset001_3DCT/nnUNetTrainer_CLSHead__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
# nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHead_Frozen -p nnUNetPlans --npz

# train classification head with sum of all scales
# echo "Starting nnU-Net v2 training..."
# export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
# export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
# export nnUNet_results="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_gd"
# export NNUNETV2_MT_NUM_CLS=3 #your number of classes
# export NNUNETV2_MT_LOSS_WEIGHT=1 #lambda for cls loss 
# export NNUNETV2_PRE_CHECKPOINT_PATH="/nfs/home/mwei/nnUNet_data/nnUNet_results/Dataset001_3DCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
# nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHeadGD -p nnUNetPlans --npz

# echo "Starting nnU-Net v2 training..."
# export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
# export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
# export nnUNet_results="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_sumOC_bs16"
# export NNUNETV2_MT_NUM_CLS=3 #your number of classes
# export NNUNETV2_MT_LOSS_WEIGHT=0.3 #lambda for cls loss 
# export NNUNETV2_PRE_CHECKPOINT_PATH="/nfs/home/mwei/nnUNet_data/nnUNet_results/Dataset001_3DCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
# nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHeadSumOC -p nnUNetPlans --npz

echo "Starting nnU-Net v2 training..."
export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
export nnUNet_results="/nfs/home/mwei/nnUNet_results/nnUNet_results_sum_headver7_bs8_focal_nopretrain"
export NNUNETV2_MT_NUM_CLS=3 #your number of classes
export NNUNETV2_MT_LOSS_WEIGHT=0.3 #lambda for cls loss 
# export NNUNETV2_PRE_CHECKPOINT_PATH="/nfs/home/mwei/nnUNet_data/nnUNet_results/Dataset001_3DCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHeadSum -p nnUNetPlans --npz


## train classification head only with simple architecture from scratch
# echo "Starting nnU-Net v2 training..."
# export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
# export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
# export nnUNet_results="/nfs/home/mwei/nnUNet_results/nnUNet_results_simpleVer2_bs12"
# export NNUNETV2_MT_NUM_CLS=3 #your number of classes
# export NNUNETV2_MT_LOSS_WEIGHT=1 #lambda for cls loss 
# # export NNUNETV2_PRE_CHECKPOINT_PATH="/nfs/home/mwei/nnUNet_data/nnUNet_results/Dataset001_3DCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
# nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHeadSimple -p nnUNetPlans --npz