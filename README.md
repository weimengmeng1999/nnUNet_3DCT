# nnUNet_3DCT
## Pancreas Lesion Segmentation and Classification with nnU-Net v2

This repository contains scripts and configurations for training and evaluating a **multi-task deep learning model** for **pancreas cancer segmentation and classification** from 3D CT scans.  
The approach extends **[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)** with a **classification head** for lesion subtype prediction.  
We explore methods to improve classification accuracy and inference efficiency

![Model Architecture](figures/model_3dct.png)


## Setup

Follow the **[Dockerfile](nnunet_3dct/Dockerfile)** or using the following for the installation. 

```bash
cd nnunet_3dct
pip install -e .
```

Before running training or inference, set the following environment variables:

```bash
cd nnunet_3dct
pip install -e .

export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

Preprocess and check dataset integrity:

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Note: if you use T4 GPU on colab, this will give you batch size 3 for your plans.json, which could influence the training. We suggest to train the model with A100 GPU for batch size > 8.

## Pre-trained Weights

Download the pre-trained weights here for stage 1 (segmentation only):
[pretrained_model_stage1.pth](https://drive.google.com/file/d/14dNt-dVwrSkov8c3B4TobnK9CUhWGRYv/view?usp=sharing)

Download the pre-trained weights here for stage 2 (multi-task: segmentation and classification):
[pretrained_model_stage2.pth](https://drive.google.com/file/d/13tbgtirXdxdkgwsDelrkoXT6E5GQTKTa/view?usp=sharing)

## Training

Follow the instructions or use **[train.sh](train.sh)** to conduct training.

### Stage 1: Train Segmentation Only
Only train segmentation branch using lesion and pancreas masks as supervision.

```bash
nnUNetv2_train Dataset001_3DCT 3d_fullres 0 --npz
```

### Stage 2: Train Jointly with Classification

#### Fine-tune from stage 1 Pre-trained

Fin-tune the classification head and original nnUNet network (encoder + segmentation head) from the pre-trained weights that trained only for segmentation.

```bash
export NNUNETV2_MT_NUM_CLS=3 #your number of classes
export NNUNETV2_MT_LOSS_WEIGHT=0.3 #lambda for cls loss 
#IMPORTANT: load the pre-trained weights from stage 1 
export NNUNETV2_PRE_CHECKPOINT_PATH="/path/to/checkpoint_best_stage1.pth" 
nnUNetv2_train Dataset001_3DCT 3d_fullres 0 -tr nnUNetTrainer_CLSHeadSumFT -p nnUNetPlans --npz
```

#### Train from Scratch

Train the segmentation head with classification head together from scratch.

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=0.3

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSum -p nnUNetPlans --npz
```

### Train Varient Classification Head

**Train classification head only with single-scale (encoder bottleneck feature) architecture (from scratch)**

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=1

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSimple -p nnUNetPlans --npz
```

**Train classification head and segmentation head with different optimizer (from scratch)**

```bash
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=1

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSepOpt -p nnUNetPlans --npz
```
## Inference

Follow the instructions or use **[inf.sh](inf.sh)** to conduct training.

### Stage 1: Segmentation only
Inference original nnUNet version.

```bash
nnUNetv2_predict \
  -i /path/to/imagesVal \
  -o /path/to/output_segmentation \
  -d Dataset001_3DCT \
  -c 3d_fullres \
  -tr nnUNetTrainer_CLSHeadSumFT \ #only the name for checkpoint folder
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_best.pth \
  --save_probabilities
```

### Stage 2: Segmentation + Classification
Inference with improved speed.

```bash
python nnunetv2/inference/predict_classification_fast_CLSHeadSum.py \
  -i /path/to/imagesVal \
  -o /path/to/output \
  -co /path/to/output \
  -d Dataset001_3DCT \
  -c 3d_fullres \
  -tr nnUNetTrainer_CLSHeadSumFT \
  -p nnUNetPlans \
  -f 0 \
  -chk checkpoint_latest.pth \
  --save_probabilities
```

## Evaluation

Follow the instructions or use **[eval.sh](eval.sh)** to conduct training.

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

## Evaluation Results

The classification performance report can be found in **[classification metrics report](stage2_output/classification_metrics.json)**.

The segmentation performance for multi-class can be found in **[pancreas/lesion segmentation report](stage2_output/summary.json)**.

The segmentation performance for whole pancreas can be found in **[whole pancreas segmentation report](stage2_output/whole_pancreas_summary.json)**.

## Quickstart (End-to-End Example)

```bash
# 1. Set paths
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results_sum_bs8

# 2. Train model for stage 1 (segmentation only)

nnUNetv2_train Dataset001_3DCT 3d_fullres 0 --npz

# 3. Train model (fine-tune with clalssification branch)
export NNUNETV2_MT_NUM_CLS=3
export NNUNETV2_MT_LOSS_WEIGHT=0.3
export NNUNETV2_PRE_CHECKPOINT_PATH="/path/to/checkpoint_best_stage1.pth" 
nnUNetv2_train Dataset001_3DCT 3d_fullres 0 \
  -tr nnUNetTrainer_CLSHeadSumFT -p nnUNetPlans --npz

# 4. Fast inference (segmentation + classification)
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

# 5. Evaluate segmentation (pancreas/lesion/background)
nnUNetv2_evaluate_folder \
  $nnUNet_raw/Dataset001_3DCT/labelsVal \
  $nnUNet_results/segreTr_fast \
  -djfile $nnUNet_raw/Dataset001_3DCT/dataset.json \
  -pfile $nnUNet_results/Dataset001_3DCT/nnUNetTrainer_CLSHeadSum__nnUNetPlans__3d_fullres/fold_0/plans.json

# 6. Evaluate whole-pancreas segmentation performance
python nnunetv2/evaluation/evaluate_whole_pancreas.py \
  $nnUNet_raw/Dataset001_3DCT/labelsVal \
  $nnUNet_results/segreTr_fast \
  -djfile $nnUNet_raw/Dataset001_3DCT/dataset.json

# 7. Evaluate classification
python nnunetv2/evaluation/evaluate_classification.py \
  $nnUNet_results/segreTr_fast \
  $nnUNet_raw/Dataset001_3DCT/classification.json \
  --class-names "subtype0" "subtype1" "subtype2"
```
```


## Notes

- Batch size is important for training the classification head. We highly recommend to train for >8 bs.
- Adjust `NNUNETV2_MT_NUM_CLS` and `NNUNETV2_MT_LOSS_WEIGHT` depending on the number of classes and classification weight in your task.
- For fine-tuning classification with segmentation training, a pre-trained segmentation model checkpoint must be provided via `NNUNETV2_PRE_CHECKPOINT_PATH`.

