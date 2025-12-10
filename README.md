# Optimizing BERT-base Fake News Classification with LoRA and Mixed Precision

This repository contains the source code and experimental results for the project **"Optimizing BERT-base Fake News Classification with LoRA and Mixed Precision"**.

We investigate high-performance training strategies for BERT-based fake news classification on the LIAR dataset. We systematically compare three configurations:

1.  **Full-Precision Fine-Tuning (FP32)**
2.  **Automatic Mixed Precision (AMP)**
3.  **Low-Rank Adaptation (LoRA) + AMP**

Our goal is to achieve competitive Macro-F1 scores while significantly reducing training time and GPU memory usage through mixed precision and parameter-efficient fine-tuning (PEFT).

-----

## 1\. Project Description

Automatic fact-checking is a high-stakes domain requiring efficient NLP models. In this project, we utilize the **LIAR dataset**, converting the original six-way classification into a binary task (Fake vs. True).

Key features of this project include:

  * **Model Architecture**: `bert-base-uncased` with a linear classification head.
  * **Optimization**: PyTorch AMP (Automatic Mixed Precision) for memory reduction and speedup.
  * **PEFT**: LoRA (Low-Rank Adaptation) applied to Query/Value projections to reduce trainable parameters.
  * **Data Handling**: Class reweighting to handle label imbalance and rich metadata textualization (Speaker, Job, Context, etc.).
  * **Profiling**: Deep dive into CUDA operator-level performance (GEMM vs. Attention kernels) using PyTorch Profiler.
  * **Threshold Tuning**: Validation-based threshold optimization to maximize Macro-F1.

-----

## 2\. Project Milestones

| Milestone | Description | Status |
| :--- | :--- | :---: |
| **Dataset Preparation** | Data loading via KaggleHub, text preprocessing, and binary label conversion. |  Completed |
| **Baseline Implementation** | Full fine-tuning of BERT in FP32 with class reweighting. |  Completed |
| **AMP Integration** | Implementation of `torch.amp` for mixed-precision training. |  Completed |
| **LoRA Implementation** | Integration of PEFT library for parameter-efficient fine-tuning. |  Completed |
| **Hyperparameter Tuning** | Grid search for LoRA rank, alpha, and dropout. |  Completed |
| **Performance Profiling** | Operator-level analysis (CPU/CUDA time, Memory) using PyTorch Profiler. |  Completed |
| **Threshold Optimization** | Post-processing technique to optimize decision thresholds for unbalanced classes. |  Completed |
| **Final Evaluation** | Comprehensive comparison of Accuracy, Macro-F1, Speed, and Memory usage. |  Completed |

-----

## 3\. Repository and Code Structure

The entire project logic is encapsulated in a single Jupyter Notebook: `HPML_Project.ipynb`.

### File Structure

```text
.
├── HPML_Project.ipynb    # Main execution notebook containing all experiments
├── README.md             # Project documentation
└── requirements.txt      # (Optional) Dependencies list
```

### Notebook Sections Overview

The `HPML_Project.ipynb` is organized into the following logical blocks:

1.  **Environment Setup**: Helper functions for GPU memory tracking and Dataset class definition (`TextualizedLIARDataset`).
2.  **Profiler Utility**: A custom `train_epoch_with_profiler` function to capture CUDA kernel statistics for the first 5 batches.
3.  **Baseline + AMP**:
      * Full fine-tuning using `torch.amp.autocast`.
      * Includes training loop, validation, and profiling export.
4.  **LoRA + AMP**:
      * Grid search implementation for finding optimal LoRA hyperparameters ($r, \alpha, dropout$).
      * Final training loop using the best configuration.
5.  **Baseline (FP32)**:
      * Standard full-precision training for benchmarking purposes.
6.  **Threshold Optimization**:
      * Logic to sweep thresholds $[0.1, 0.9]$ on validation logits to maximize F1 score before testing.

-----

## 4\. How to Execute

### Prerequisites

The code is designed to run in a GPU-accelerated environment (e.g., Google Colab, Kaggle Notebooks, or a local server with NVIDIA GPU).

**Dependencies:**

```bash
pip install torch transformers scikit-learn pandas peft kagglehub
```

### Running the Code

1.  **Download the Notebook**: Clone this repo or download `HPML_Project.ipynb`.
2.  **Dataset Access**: The code uses `kagglehub` to automatically download the LIAR dataset.
      * *Note: You may need to authenticate with Kaggle if the dataset is private, though the code assumes public access.*
3.  **Run All Cells**: Execute the cells sequentially. The notebook will:
      * Download the data.
      * Train the **Baseline+AMP** model and save `best_baselineamp_model.pth`.
      * Run **LoRA** Grid Search, train the best LoRA model, and save `best_lora_model.pth`.
      * Train the **FP32 Baseline** model and save `best_baselineonly_model.pth`.
      * Output classification reports and profiler tables for each run.

### Viewing Profiler Traces

The code exports Chrome traces (e.g., `profiler_trace_epoch1.json`). To view them:

1.  Download the `.json` file generated in the notebook directory.
2.  Open Google Chrome and navigate to `chrome://tracing`.
3.  Load the `.json` file to visualize the GPU timeline.

-----

## 5\. Results and Observations

We evaluated three configurations on a single NVIDIA A100 GPU.

### A. Classification Performance

*Threshold calibration was applied to maximize Macro-F1.*

| Method | Accuracy | Macro F1 | Fake Recall | True Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (FP32)** | 0.66 | 0.65 | 0.54 | 0.76 |
| **Baseline + AMP** | 0.65 | 0.64 | 0.59 | 0.70 |
| **LoRA + AMP** | 0.61 | 0.60 | 0.52 | 0.68 |

### B. Computational Efficiency

*AMP significantly reduces memory and time, while LoRA provides further memory savings.*

| Method | Trainable Params | Avg Epoch Time (s) | Peak Memory (MB) |
| :--- | :---: | :---: | :---: |
| **Baseline (FP32)** | 109.5M | 78.5s | \~20,878 MB |
| **Baseline + AMP** | 109.5M | 14.7s | \~16,057 MB |
| **LoRA + AMP** | 0.30M | 13.5s | \~11,494 MB |

### C. Key Observations

1.  **Speedup**: Mixed Precision (AMP) delivers a **\>5x speedup** compared to FP32 by utilizing Tensor Cores for matrix multiplications.
2.  **Memory**: LoRA + AMP creates the smallest footprint (\~11.5 GB), reducing memory usage by **\~45%** compared to the FP32 baseline.
3.  **Profiling Insights**:
      * **GEMM (Linear Layers)**: AMP significantly accelerates these operations. LoRA further reduces the cost of projection layers by reducing the rank of trainable matrices.
      * **Attention Bottleneck**: While LoRA optimizes linear projections, the **Attention mechanism** remains a computational bottleneck (runtime dominated by sequence length), which LoRA does not directly optimize.
4.  **Trade-off**: LoRA + AMP offers the best efficiency for resource-constrained environments, though it incurs a slight drop in Macro-F1 compared to full fine-tuning.

-----

### Authors

  * Dawei Sun
  * Yikai Xu