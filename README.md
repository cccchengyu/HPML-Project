# HPML Project: Fake News Detection with BERT and LoRA

## Overview
This project implements a high-performance fake news detection system using BERT models with various optimization techniques. The system is trained on the LIAR dataset and explores different training strategies including baseline models, mixed precision training (AMP), and parameter-efficient fine-tuning with LoRA.

**Platform**: This project is designed to run on **Google Colab** with GPU acceleration (A100 recommended).

## Dependencies and Libraries

### Core Libraries
- **PyTorch**: Deep learning framework for model training and inference
- **Transformers (Hugging Face)**: Pre-trained BERT models and tokenizers
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA implementation for efficient model adaptation

### Data Processing
- **Pandas**: Dataset loading and manipulation
- **NumPy**: Numerical operations and array handling
- **Kagglehub**: Dataset download from Kaggle

### Evaluation and Metrics
- **Scikit-learn**: Model evaluation metrics (accuracy, F1-score, classification report)

### Performance Monitoring
- **PyTorch Profiler**: Detailed profiling of training operations
- **CUDA Memory Tools**: GPU memory usage tracking

## How to Run the Project

### Prerequisites
- A Google account to access Google Colab
- Recommended: Colab Pro or Pro+ for access to A100 GPU
- Ensure GPU runtime is enabled: Runtime → Change runtime type → Hardware accelerator → GPU

### 1. Essential Setup Cells

Run these foundational cells first in order:

#### **Import**
Loads all necessary libraries and dependencies for the project.

#### **Time and Memory**
Provides utility functions to track training time and GPU memory consumption throughout the experiments.

#### **Load Dataset**
Downloads and prepares the LIAR dataset from Kaggle, setting up train, validation, and test splits.

#### **Profiler Function**
Implements a profiling mechanism that monitors the first 5 batches of the second epoch during training, capturing detailed performance metrics for analysis.

---

### 2. Model Training Options

The project offers three training configurations, each with different optimization strategies:

#### **Only Baseline**
- Standard BERT training without optimizations
- After training, run **Rebuild for Baseline Only** to optimize the classification threshold

#### **Baseline + AMP**
- BERT training with Automatic Mixed Precision for faster computation
- After training, run **Rebuild for Baseline + AMP** to optimize the classification threshold

#### **Model with LoRA + AMP**  Special Instructions
This option requires additional setup steps:

1. **LoRA Cell**: Run this first to download LoRA-related components
2. **Grid Search Function**: Define the hyperparameter search space
3. **LoRA Grid Search Before Final Training**: Execute grid search to find optimal parameters
4. **Get Best Params**: Extract the best hyperparameter configuration from grid search results
5. **Model with LoRA + AMP**: Train the final model with optimized parameters
6. **Rebuild for LoRA + AMP**: Optimize the classification threshold for the trained model

---

### 3. Threshold Optimization

Due to class imbalance in the LIAR dataset (unequal distribution of True/False labels), each model requires threshold tuning to maximize performance. The **Rebuild** cells perform this optimization by:
- Evaluating multiple threshold values on the validation set
- Selecting the threshold that maximizes macro F1-score
- Applying the optimized threshold to test set predictions

---

## Execution Order Summary

### For Baseline Models:
```
Import → Time and Memory → Load Dataset → Profiler Function 
  → Only Baseline → Rebuild for Baseline Only
  OR
  → Baseline + AMP → Rebuild for Baseline + AMP
```

### For LoRA Model:
```
Import → Time and Memory → Load Dataset → Profiler Function 
  → LoRA Cell → Grid Search Function 
  → LoRA Grid Search Before Final Training → Get Best Params 
  → Model with LoRA + AMP → Rebuild for LoRA + AMP
```

---

## Notes
- **Platform Requirements**: This notebook is optimized for Google Colab environment with GPU acceleration
- Each model's training cell can be run independently after completing the setup cells
- The profiling data helps identify performance bottlenecks in different training configurations
- Grid search for LoRA significantly improves model performance by finding optimal hyperparameters
- Threshold optimization is crucial for handling the imbalanced dataset effectively
- Runtime may vary depending on the GPU type allocated by Colab (A100 > V100 > T4)
