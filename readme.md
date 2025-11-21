# Reproduction of GraphDTA on Davis and KIBA Datasets under Cold-Start Settings

## Introduction

This project reproduces GraphDTA on the Davis and KIBA datasets under various cold-start settings.

Author: Dun Zihan  
Date: 2025/11/19

## Environment Setup

### Dependencies

- Python 3.9  
- PyTorch 2.0.0  
- Other required libraries

## Running Instructions

### Dataset Description

The directory `./data/oursplit` contains:

- {dataset_name}_train_canonical.csv  
- {dataset_name}_test_canonical.csv  
- {dataset_name}_test1_canonical.csv  
- {dataset_name}_test2_canonical.csv  

Here, `{dataset_name}` refers to `KIBA` or `Davis`.  

- `train` is the training set  
- `val` is the val set  
- `test` is the unseen-pair set  
- `test1` is the unseen-drug set  
- `test2` is the unseen-target set  

Training Set: Contains 80% of drugs and 80% of targets,
along with their corresponding interaction pairs. This set
is used to train the model.  
Validation and Test Sets: Constructed from the remaining
20% of drugs and targets to simulate cold-start conditions.
We define three disjoint evaluatoin settings:

- Unseen Drugs (UD): Comprises interactions between
20% new drugs and 80% training targets.  
- Unseen Targets (UT): Comprises interactions be-
tween 20% new targets and 80% training drugs.  
- Unseen Pairs (UP): Comprises interactions between
20% new drugs and 20% new targets, forming en-
tirely novel drug-target combinations.

### Execution Steps

1. **Preprocess the dataset**

```python
   python create_data.py
```

2. **Train and evaluate the model**

```python
   python training.py
```

## Experimental Results

### Davis

| Settings | MSE(std) | MAE(std) | CI(std) | R^2  |
|-------|-------|-------|-------|-------|
| Unseen-Drug | 0.590(0.016) | 0.552(0.014) | 0.695(0.008) | 0.053 |
| Unseen-Target | 0.565(0.031) | 0.527(0.012) | 0.763(0.018) | 0.399 |
| Unseen-Pair | 0.874(0.036) | 0.606(0.023) | 0.631(0.021) | 0.032 |

### KIBA

| Settings | MSE(std) | MAE(std) | CI(std) | R^2   |
|-------|-------|-------|-------|-------|
| Unseen-Drug | 0.400(0.011) | 0.444(0.016) | 0.752(0.004) | 0.444 |
| Unseen-Target | 0.457(0.045) | 0.481(0.027) | 0.627(0.028) | 0.215 |
| Unseen-Pair | 0.551(0.054) | 0.536(0.030) | 0.574(0.023) | 0.045 |
