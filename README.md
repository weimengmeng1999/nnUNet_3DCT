# nnUNet_3DCT
## Pancreas Lesion Segmentation and Classification with nnU-Net v2

This repository contains scripts and configurations for training and evaluating a **multi-task deep learning model** for **pancreas cancer segmentation and classification** from 3D CT scans.  
The approach extends **[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)** with a **classification head** for lesion subtype prediction.  
We explore methods to improve classification accuracy and inference efficiency.


## Setup

Before running training or inference, set the following environment variables:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

Preprocess and check dataset integrity:

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Note: if you use T4 GPU on colab, this will give you batch size 3 for your plans.json, which could influence the training. We suggest to train the model with A100 GPU for batch size > 8.

## Training

Follow the instructions or use **[train.sh](train.sh)** to conduct training.

### Train from Scratch

Train the segmentation head with classification head together from scratch.

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=0.3

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSum -p nnUNetPlans --npz
```

### Fine-tune from Pre-trained

Fin-tune the classification head and original nnUNet network (encoder + segmentation head) from the pre-trained weights that trained only for segmentation.

```bash
export NNUNETV2_MT_NUM_CLS=3 #your number of classes
export NNUNETV2_MT_LOSS_WEIGHT=0.3 #lambda for cls loss 
export NNUNETV2_PRE_CHECKPOINT_PATH="/path/to/checkpoint_best.pth"
nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHeadSumFT -p nnUNetPlans --npz
```
### Train 

### Examples

**Train segmentation + classification jointly**

```bash
export NNUNETV2_MT_NUM_CLS=3   # number of classes
export NNUNETV2_MT_LOSS_WEIGHT=1   # λ for classification loss

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHead -p nnUNetPlans --npz
```

**Train segmentation + classification with dropout**

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=0.05

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadDP -p nnUNetPlans --npz
```

**Train classification head only (frozen encoder)**

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=1
export NNUNETV2_PRE_CHECKPOINT_PATH="/path/to/checkpoint_best.pth"

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHead_Frozen -p nnUNetPlans --npz
```

**Train classification head with sum of all scales**

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=1
export NNUNETV2_PRE_CHECKPOINT_PATH="/path/to/checkpoint_best.pth"

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadGD -p nnUNetPlans --npz
```

**Train classification head with output channel sum (SumOC)**

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=0.3
export NNUNETV2_PRE_CHECKPOINT_PATH="/path/to/checkpoint_best.pth"

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSumOC -p nnUNetPlans --npz
```

**Train classification head with sum pooling (Sum Head v7)**

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=0.3

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSum -p nnUNetPlans --npz
```

**Train classification head only with simple architecture (from scratch)**

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=1

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSimple -p nnUNetPlans --npz
```

## Inference

### Segmentation only

```bash
nnUNetv2_predict \
  -i /path/to/imagesVal \
  -o /path/to/output_segmentation \
  -d Dataset001_3DCT \
  -c 3d_fullres \
  -tr nnUNetTrainer_CLSHead \
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_best.pth \
  --save_probabilities
```

### Segmentation + Classification

**Slow version:**

```bash
python nnunetv2/inference/predict_classification.py \
  -i /path/to/imagesVal \
  -o /path/to/output \
  -co /path/to/output \
  -d Dataset001_3DCT \
  -c 3d_fullres \
  -tr nnUNetTrainer_CLSHead \
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_best.pth \
  --save_probabilities
```

**Fast version:**

```bash
python nnunetv2/inference/predict_classification_fast_CLSHeadSum.py \
  -i /path/to/imagesVal \
  -o /path/to/output \
  -co /path/to/output \
  -d Dataset001_3DCT \
  -c 3d_fullres \
  -tr nnUNetTrainer_CLSHeadSum \
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_latest.pth \
  --save_probabilities
```

## Evaluation

### Segmentation evaluation

```bash
nnUNetv2_evaluate_folder \
  /path/to/labelsVal \
  /path/to/predictions \
  -djfile /path/to/dataset.json \
  -pfile /path/to/plans.json
```

### Whole-pancreas evaluation

```bash
python nnunetv2/evaluation/evaluate_whole_pancreas.py \
  /path/to/labelsVal \
  /path/to/predictions \
  -djfile /path/to/dataset.json
```

### Classification evaluation

```bash
python nnunetv2/evaluation/evaluate_classification.py \
  /path/to/predictions \
  /path/to/classification.json \
  --class-names "subtype0" "subtype1" "subtype2"
```

## Quickstart (End-to-End Example)

```bash
# 1. Set paths
export nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw"
export nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed"
export nnUNet_results="/nfs/home/mwei/nnUNet_results/nnUNet_results_sum_bs8"

# 2. Train model (sum head classification)
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=0.3

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSum -p nnUNetPlans --npz

# 3. Fast inference (segmentation + classification)
python nnunetv2/inference/predict_classification_fast_CLSHeadSum.py \
  -i $nnUNet_raw/Dataset001_3DCT/imagesVal \
  -o $nnUNet_results/segreTr_fast \
  -co $nnUNet_results/segreTr_fast \
  -d Dataset001_3DCT \
  -c 3d_fullres \
  -tr nnUNetTrainer_CLSHeadSum \
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_latest.pth \
  --save_probabilities

# 4. Evaluate classification
python nnunetv2/evaluation/evaluate_classification.py \
  $nnUNet_results/segreTr_fast \
  $nnUNet_raw/Dataset001_3DCT/classification.json \
  --class-names "subtype0" "subtype1" "subtype2"
```


## Notes

- Adjust `NNUNETV2_MT_NUM_CLS` and `NNUNETV2_MT_LOSS_WEIGHT` depending on the number of classes and classification weight in your task.
- For classification-only training, a pre-trained segmentation model checkpoint must be provided via `NNUNETV2_PRE_CHECKPOINT_PATH`.

