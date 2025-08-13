export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
export nnUNet_results="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_ocls_newmodel_dpreduce"
# infer for raw data prediction
# nnUNetv2_evaluate_folder \
# /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/labelsVal \
# /nfs/home/mwei/nnUNet_3d_data/segreVal_wocls \
# -djfile /nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed/Dataset001_3DCT/dataset.json \
# -pfile /nfs/home/mwei/nnUNet_3d_data/nnUNet_results/Dataset001_3DCT/nnUNetTrainer_CLSHead__nnUNetPlans__3d_fullres/plans.json

#evaluate whole pancreas
# python nnunetv2/evaluation/evaluate_whole_pancreas.py \
#   /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/labelsVal \
#   /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wcls2LN/segreVal_slow \
#   -djfile /nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed/Dataset001_3DCT/dataset.json

#evaluate classification
python nnunetv2/evaluation/evaluate_classification.py \
  /nfs/home/mwei/nnUNet_3d_data/nnUNet_results_ocls_newmodel_dpreduce/segreVal_slow \
  /nfs/home/mwei/nnUNet_3d_data/nnUNet_raw/Dataset001_3DCT/classification.json \
  --class-names "subtype0" "subtype1" "subtype2"