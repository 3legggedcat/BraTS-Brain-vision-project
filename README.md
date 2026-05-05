# How To Run

This folder contains:

- code:
  - `train_resnet152_unet.py`
  - `analyze_brats_project.py`
  - `analyze_brats_data_only.py`
  - `generate_preprocessing_report.py`
- dataset(Must download, too big of a file for github):
  - `BraTS2020_TrainingData/`
  - `BraTS2020_ValidationData/`
- dependencies:
  - `requirements.txt`
## Download Dataset
[Brats20 Dataset (Training + Validation)](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

## 1. Set up
Create a Python environment

```bash
python -m venv .venv
```

Activate the environment
```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 2. Train the model

### Train on 1 patient for 1 epoch

```bash
python3 train_resnet152_unet.py --max-patients 1 --epochs 1 --batch-size 2 --num-workers 0
```

### Train on 1 patient for 2 epochs

```bash
python3 train_resnet152_unet.py --max-patients 1 --epochs 2 --batch-size 2 --num-workers 0
```

### Train on 2 patients for 1 epoch

```bash
python3 train_resnet152_unet.py --max-patients 2 --epochs 1 --batch-size 2 --num-workers 0
```

### Train on 2 patients for 2 epochs

```bash
python3 train_resnet152_unet.py --max-patients 2 --epochs 2 --batch-size 2 --num-workers 0
```

### Save the checkpoint with the expected name

If you want `analyze_brats_project.py` to evaluate the trained model, save the checkpoint as:

```bash
python3 train_resnet152_unet.py --max-patients 2 --epochs 2 --output resnet152_unet_brats2020.pt
```

You can combine that with any settings you want, for example:

```bash
python3 train_resnet152_unet.py --max-patients 1 --epochs 1 --output resnet152_unet_brats2020.pt
```

## 3. Get accuracy-style scores for background and classes 1, 2, 3

Run:

```bash
python3 analyze_brats_project.py
```

If the checkpoint file `resnet152_unet_brats2020.pt` exists, the script prints a line like:

```text
per_class_hard_dice=background:0.____, label1:0.____, label2:0.____, label4_as_3:0.____
```

Read it like this:

- `background` = background score
- `label1` = class 1 score
- `label2` = class 2 score
- `label4_as_3` = class 3 score

Important:

- BraTS label `4` is remapped to class `3` in this project.
- Because of that, class `3` is printed as `label4_as_3`.
- These values are Dice scores, which are the class-by-class segmentation accuracy metric used here.

## 4. Dataset-only analysis

Run:

```bash
python3 analyze_brats_data_only.py
```

This gives dataset statistics only. It does not evaluate a trained checkpoint.

## 5. Preprocessing report

Run:

```bash
python3 generate_preprocessing_report.py --max-analysis-patients 24
```

## 6. Help

See all training options:

```bash
python3 train_resnet152_unet.py --help
```
