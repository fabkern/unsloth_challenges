# **QLoRA + FSDP2 + Single-Kernel Decode + `torch.compile` – Unsloth Challenge B**

## **📌 Overview**
This repository contains my solution for **Unsloth’s Challenge B**, which involves **finetuning LLaMA 3.1 8B on 2× GPUs with FSDP2 and QLoRA**.  
- ✅ **Fully compatible with Hugging Face’s `Trainer` and `SFTTrainer`.**  
- ✅ **Uses FSDP2 to efficiently shard QLoRA LoRA-adapted weights across GPUs.**  
- ✅ **Implements a Triton-based **single-kernel NF4 dequantization**, faster than naive decode.**  
- ✅ **Uses `torch.compile()` to optimize LoRA modules selectively.**  
- ✅ **Runs in a free Kaggle notebook with 2× Tesla T4 GPUs.**  

⚠ **Pipeline parallelism with `zero bubble scheduling` is implemented but disabled by default** due to **Kaggle’s ephemeral GPU environment** potentially breaking execution.

---

## **🚀 Features Implemented**
This script **fully implements** all required challenge components while optimizing performance.

| **Feature** | **Status** | **Implementation Proof** |
|------------|-----------|--------------------------|
| **Finetunes LLaMA 3.1 8B with FSDP2 on 2 GPUs** | ✅  | Uses `fully_shard` and `auto_wrap_policy` for `LlamaDecoderLayer`. |
| **QLoRA (4-bit NF4, frozen base, LoRA adapters added)** | ✅  | Uses `get_peft_model()` and `LoraConfig`. |
| **Mixed Precision (FP16) & CPU Offload in FSDP2** | ✅  | Configured with `MixedPrecisionPolicy` and `CPUOffloadPolicy`. |
| **Fully compatible with `Trainer` / `SFTTrainer`** | ✅  | Uses `TrainingArguments` inside `SFTTrainer`. |
| **Pipeline Parallelism (`zero bubble scheduling`)** | ⚠ **Implemented but Disabled** | Implemented in `create_pipeline_model()`, disabled in Kaggle. |
| **Selective `torch.compile()` for LoRA submodules** | ✅  | Calls `compile_lora_modules()` to optimize LoRA layers. |
| **Minimal dataset for fast Kaggle execution** | ✅  | Uses `"train[:100]"` to **avoid ephemeral kills**. |
| **Triton-based single-kernel NF4 decode (faster than naive)** | ✅  | Implements `decode_kernel_fp16()` vs. `naive_decode_nf4()` and measures speed. |

---

## **🛠️ Implementation Details**
### **🔹 Finetuning LLaMA 3.1 8B with QLoRA + FSDP2**
- Loads **pre-quantized NF4 4-bit LLaMA 3.1 8B** with **bitsandbytes**.
- Applies **QLoRA**, freezing all base model weights.
- Uses **FSDP2** to shard **only LoRA parameters**, distributing training across **2 GPUs**.
- Converts **frozen integer parameters to buffers** to **avoid unnecessary FSDP overhead.**
- Enables **mixed precision (`fp16`) and CPU offloading**.

### **🔹 Pipeline Parallelism with Zero Bubble Scheduling**
- **Implemented in `create_pipeline_model()` but DISABLED by default** to prevent execution failures in **Kaggle’s ephemeral environment**.
- If enabled, splits LLaMA layers across **two pipeline stages**, reducing communication overhead.
- Uses **`ScheduleInterleavedZeroBubble()`** to overlap **compute and communication**.

⚠ **To enable pipeline parallelism, set:**
```python
os.environ["USE_PIPELINE"] = "1"
```
However, **due to ephemeral memory kills on Kaggle, this is disabled**.

### **🔹 `torch.compile()` Optimization for LoRA Modules**
- **Compiles only LoRA submodules** instead of the entire model.
- **Why?** LLaMA 3.1 8B is **too large** for `torch.compile()` to optimize in full.
- Calls **`compile_lora_modules()`** to selectively **compile only LoRA layers**, reducing compilation overhead.

### **🔹 Minimal Dataset (Avoiding Ephemeral Kills)**
- Uses **a minimal dataset slice (`train[:100]`)** to **avoid long execution times**.
- This ensures **the training process doesn’t get killed mid-run**.

### **🔹 Triton-Based Single-Kernel NF4 Decode**
- Implements **Triton single-kernel NF4 dequantization** (`decode_kernel_fp16()`).
- Compares against **naive decode** (`naive_decode_nf4()`).
- **Speedup is measured** to **prove single-kernel execution is faster**.

---

## **🖥️ Running This in Kaggle (2× T4 GPUs)**
This script is **designed to run in Kaggle** with **2× Tesla T4 GPUs**.

### **📌 Steps to Run**
1. **Ensure your Kaggle account has access to Hugging Face models.**
2. **In Kaggle, go to "Settings" → Enable "Accelerator = 2 GPUs".**
3. **Ensure your Hugging Face token is stored in `Kaggle Secrets`** (`HF_TOKEN`).
4. **Run this script**.

### **💾 Expected Output**
- **Successfully loads LLaMA 3.1 8B NF4 in QLoRA mode**.
- **FSDP2 successfully shards and trains LoRA adapters**.
- **Training loss curve is equivalent to single-GPU training**.
- **Triton kernel executes single-pass NF4 decode faster than naive.**
- **Pipeline parallelism (`zero bubble scheduling`) is optionally available** but is disabled because Kaggle’s ephemeral setup may break.

## 🔗 **Solution Links**
- **Kaggle Notebook (QLoRA + FSDP2 on 2x T4 GPUs):** [Click Here](https://www.kaggle.com/code/rootyo/notebook-challenge-b)
