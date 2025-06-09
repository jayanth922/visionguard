# Real-Time Image Corruption Detection for Autonomous Vehicles

This repository contains an end-to-end Kaggle-based pipeline for detecting sensor-feed corruption in autonomous-vehicle camera data. All steps run entirely in the cloud—no local GPU or special hardware required—and culminate in a live demo hosted as a Hugging Face Space.

---

## ✔️ What We’ve Implemented So Far

### Phase 1: Environment & Data Setup  
- Created a GPU-enabled Kaggle notebook environment.  
- Mounted and inspected the `klemenko/kitti-dataset` (images, calibration, labels).  
- Built a lightweight project folder structure under `/kaggle/working/`.

### Phase 2: Data Parsing & Splitting  
- Generated 70/15/15 train/val/test CSV manifest files pointing at the Kaggle input.  
- Verified counts and sample file paths.

### Phase 3: Dataset & Model Training  
- Defined a `CorruptionDataset` with three modes:
  - **clean** (label 0),
  - **augmix** (label 1, using TimM’s AugMix),
  - **corrupt** (label 1, using Albumentations real-world distortions).  
- Built PyTorch DataLoaders with mixed batches for all modes.  
- Fine-tuned a pretrained ResNet-18 head for binary classification (clean vs. corrupt).  
- Enabled mixed-precision training, TensorBoard logging, checkpointing, and LR scheduling.  
- Trained for 20 epochs, achieving high val accuracy and saving the best model.

### Phase 4: Robustness Benchmarking  
- Implemented an `evaluate(...)` function that:
  - Computes accuracy, precision, recall, F1, false-negative and false-positive rates.
  - Forces 2×2 confusion matrices even when one class is absent.  
- Benchmarked on:
  - **Clean test set** (measured FPR ≈ 2–5 %)  
  - **Albumentations-corrupted test set** (≈ 99.8 % accuracy, <0.2 % FNR)  
- Saved results to `reports/benchmark_results.json`.

### Phase 5: ONNX Export & Optimization  
- Reconstructed the ResNet-18 model in PyTorch and loaded the best checkpoint.  
- Exported to ONNX (`visionguard.onnx`) with dynamic batch sizing.  
- Simplified the ONNX graph (`visionguard_simplified.onnx`) via `onnxsim`.  
- Verified bitwise equivalence (max diff = 0).  
- Measured average GPU inference latency (~ < 10 ms) with `onnxruntime-gpu`.

### Phase 6: Cloud Deployment as a Hugging Face Space  
- Created a **public** Hugging Face Space named `jayanth7111/visionguard-space`.  
- Wrote a Gradio app (`app.py`) that loads `visionguard_simplified.onnx` and:
  - Resizes inputs to 128×128,
  - Normalizes with ImageNet stats,
  - Runs inference with ONNX Runtime,
  - Returns `{clean, corrupted}` probabilities.  
- Enabled verbose (`show_error=True, debug=True`) error reporting.  
- Deployed via the Hugging Face Hub API (`HfApi.upload_file` with `repo_type="space"`).  
- Verified inference through the `gradio_client`:
  ```python
  from gradio_client import Client
  client = Client("jayanth7111/visionguard-space")
  result = client.predict({"url": data_uri}, api_name="/predict")
