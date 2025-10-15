# **Project Proposal (Final Draft)**

## **1. Project Title**

**FinBERT-LSTM with LoRA and Mixed Precision for Efficient Dow Jones Index Prediction**



## **2. Team Members (2)**

- **Student A:** Dawei Sun (ds8092@nyu.edu)
- **Student B:** Yikai Xu (yx3845@nyu.edu)



## **3. Goal / Objective**

The goal of this project is to design an **efficient and scalable multi-modal financial forecasting model** that predicts the **Dow Jones Industrial Average (DJIA)** using both **financial news headlines** and **historical market data**.

We aim to demonstrate that:

1. Incorporating **news sentiment** (via FinBERT) with **historical time-series modeling** (via LSTM) improves predictive accuracy.
2. Combining **LoRA (Low-Rank Adaptation)** and **Automatic Mixed Precision (AMP)** can significantly reduce **training time, inference latency, and model size**, without substantial accuracy loss.
3. Profiling and benchmarking can quantify these trade-offs in a reproducible, measurable way.



## **4. Challenges**

1. **Temporal alignment:** Properly aligning news and price data to avoid information leakage.
2. **Computational efficiency:** FinBERT fine-tuning is resource-heavy; we must optimize via LoRA and AMP.
3. **Model comparison:** Building fair baseline and optimized versions to measure efficiency–accuracy trade-offs.
4. **Profiling and evaluation:** Measuring latency, throughput, and GPU utilization precisely using PyTorch Profiler.



## **5. Approach / Techniques**

### **(a) Data Flow**

1. **Dataset:** *Daily News for Stock Market Prediction* (Kaggle).

   

   - Includes daily news headlines (2008–2016) and corresponding DJIA index values.

2. **Feature Extraction:**

   - Use **FinBERT** to compute sentiment probabilities [positive, negative, neutral] for daily aggregated news.
   - Retrieve historical DJIA prices (Open, High, Low, Close, Volume) for a **10-day sliding window**.

3. **Model Architecture:**

   - **Text Encoder:** FinBERT fine-tuned with LoRA adapters.
   - **Numerical Encoder:** LSTM for 10-day historical index data.
   - **Fusion Layer:** Concatenate sentiment vector and numeric embedding → dense layers → output next-day DJIA price.

4. **Training Optimizations:**

   - Fine-tune FinBERT under **LoRA** (freezing original weights).
   - Train entire model under **AMP** to accelerate computation and reduce VRAM.
   - Use **AdamW** optimizer, linear learning rate schedule, and early stopping.

### **(b) Training Targets**

- **Task:** Regression — predict next-day DJIA closing price or daily return.
- **Metrics:** RMSE, MAE, MAPE, and Directional Accuracy.



## **6. Implementation Details**

### **Hardware**

| **Component**       | **Description**                                              |
| ------------------- | ------------------------------------------------------------ |
| **Compute**         | NVIDIA A100 GPU (40GB VRAM) via NYU HPC (ECE_GY_9143 partition) |
| **Precision**       | Mixed Precision (FP16/FP32) with torch.cuda.amp              |
| **Profiling Tools** | torch.profiler                                               |
| **Deployment**      | Cloud-based (NYU HPC Compute Node)                           |



### **Software / Framework**

| **Category**                | **Software**                                    |
| --------------------------- | ----------------------------------------------- |
| Python Environment          | Python 3.10                                     |
| Deep Learning Framework     | PyTorch ≥ 2.0                                   |
| Model Fine-tuning           | transformers, peft                              |
| Data Processing             | pandas, numpy                                   |
| Financial Data Source       | Kaggle (Daily News for Stock Market Prediction) |
| Mixed Precision / Multi-GPU | accelerate                                      |
| Memory Optimization         | bitsandbytes                                    |
| Visualization               | matplotlib                                      |
| Profiling                   | torch.profiler                                  |



### **Baseline Model**

To fairly evaluate the impact of LoRA and AMP, we establish a **baseline model** identical in architecture but trained **without any efficiency techniques**.

| **Variant**          | **Finetuning**                           | **Precision**   |
| -------------------- | ---------------------------------------- | --------------- |
| **Baseline**         | Full FinBERT (all parameters trainable)  | FP32 (default)  |
| **Optimized (Ours)** | FinBERT + LoRA (r=8, α=32, dropout=0.05) | AMP (FP16/FP32) |

Both models will be trained on the same dataset splits for consistent comparison.



### **Profiling and Evaluation (Following BERT Optimization Paper)**

| **Metric**                  | **Description**                           |
| --------------------------- | ----------------------------------------- |
| **Training time / epoch**   | Time per epoch for baseline vs optimized  |
| **GPU utilization (%)**     | Average GPU usage during forward/backward |
| **Memory consumption (GB)** | Peak VRAM during training/inference       |
| **Model size (MB)**         | Serialized model checkpoint size          |
| **Inference latency (ms)**  | Average prediction time per batch         |
| **Accuracy metrics**        | RMSE, MAPE, Directional Accuracy          |

All results will be visualized in comparative bar charts and tables, reproducing the profiling structure of the referenced BERT efficiency paper.



## **7. Demo Planned**

The demo will include:

1. **Notebook walkthrough** demonstrating FinBERT sentiment extraction and LSTM training.
2. **Visualization**:
   - Daily sentiment trend vs. DJIA closing prices.
   - Predicted vs. actual price curves.
   - Profiling comparison plots (baseline vs optimized).
3. **Comparative metrics** table showing:
   - Training time reduction
   - Model size reduction
   - Inference speedup
   - Accuracy change (ΔRMSE, ΔMAPE).

## **8. References**

1. **Hu et al. (2021). “LoRA: Low-Rank Adaptation of Large Language Models.”**

   - Introduced LoRA for parameter-efficient fine-tuning (<1% trainable weights).
   - Our project applies LoRA to FinBERT to minimize GPU cost during fine-tuning.

   

2. **FinBERT-LSTM: Deep Learning based stock price prediction using News Sentiment Analysis (arXiv:2211.07392).**

   - Combined FinBERT sentiment analysis with LSTM-based forecasting.
   - We extend this by adding LoRA + AMP optimization and conducting a full profiling study.

   

3. **Financial Sentiment Analysis using FinBERT with Application in Predicting Stock Movement (arXiv:2306.02136).**

   - Validated FinBERT’s ability to capture domain-specific sentiment.
   - We use FinBERT as a foundation but move beyond classification to continuous value prediction.