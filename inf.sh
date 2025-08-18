export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
export nnUNet_results="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_sum_bs8"
# infer for raw data prediction for segmentation only
# nnUNetv2_predict \
#   -i /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/imagesVal \
#   -o /nfs/home/mwei/nnUNet_3d_data/segreVal_wocls \
#   -d Dataset001_3DCT \
#   -c 3d_fullres \
#   -tr nnUNetTrainer_CLSHead \
#   -p nnUNetPlans \
#   -f 0 \
#   -chk checkpoint_best_wocls.pth \
#   --save_probabilities   # optional, saves softmax .npz next to segmentations

# infer for raw data prediction for segmentation and classification
# python nnunetv2/inference/predict_classification.py \
#   -i /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/imagesTr \
#   -o /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wcls2LN_NBND_wf1/segreVal_slow_train \
#   -co /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wcls2LN_NBND_wf1/segreVal_slow_train \
#   -d Dataset001_3DCT \
#   -c 3d_fullres \
#   -tr nnUNetTrainer_CLSHead \
#   -p nnUNetPlans \
#   -f 0 \
#   -chk checkpoint_best.pth \
#   --save_probabilities

# infer for raw data prediction for segmentation and classification with improved inference time
# python nnunetv2/inference/predict_classification_fast.py \
#   -i /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/imagesVal \
#   -o /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wcls2LN_ver2/segreVal_fast \
#   -co /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wcls2LN_ver2/segreVal_fast \
#   -d Dataset001_3DCT \
#   -c 3d_fullres \
#   -tr nnUNetTrainer_CLSHead \
#   -p nnUNetPlans \
#   -f 0 \
#   -chk checkpoint_best.pth \
#   --save_probabilities

# classification head with DP
# python nnunetv2/inference/predict_classification.py \
#   -i /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/imagesVal \
#   -o /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wclsDP/segreVal_slow \
#   -co /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wclsDP/segreVal_slow \
#   -d Dataset001_3DCT \
#   -c 3d_fullres \
#   -tr nnUNetTrainer_CLSHeadDP \
#   -p nnUNetPlans \
#   -f 0 \
#   -chk checkpoint_best.pth \
#   --save_probabilities

# python nnunetv2/inference/predict_classification_fast.py \
#   -i /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/imagesVal \
#   -o /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wclsDP/segreVal_fast \
#   -co /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wclsDP/segreVal_fast \
#   -d Dataset001_3DCT \
#   -c 3d_fullres \
#   -tr nnUNetTrainer_CLSHeadDP \
#   -p nnUNetPlans \
#   -f 0 \
#   -chk checkpoint_latest.pth \
#   --save_probabilities

# python nnunetv2/inference/predict_classification_fast.py \
#   -i /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/imagesVal \
#   -o /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_sum_nopretrain/segreTr_fast \
#   -co /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_sum_nopretrain/segreTr_fast \
#   -d Dataset001_3DCT \
#   -c 3d_fullres \
#   -tr nnUNetTrainer_CLSHeadSum\
#   -p nnUNetPlans \
#   -f 0 \
#   -chk checkpoint_best.pth \
#   --save_probabilities

python nnunetv2/inference/predict_classification_fast_CLSHeadSum.py \
  -i /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/imagesVal \
  -o /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_sum_bs8/segreTr_fast \
  -co /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_sum_bs8/segreTr_fast \
  -d Dataset001_3DCT \
  -c 3d_fullres \
  -tr nnUNetTrainer_CLSHeadSum\
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_latest.pth \
  --save_probabilities
