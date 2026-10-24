# Deep Learning Anomaly Detection Framework

A production-ready deep learning implementation designed to detect structural anomalies and out-of-distribution variations in high-dimensional system telemetry data using an **Autoencoder Neural Network** architecture.

## 🚀 Architectural Overview

Autoencoders are neural networks trained to copy their inputs to their outputs. They contain a tight hidden bottleneck layer that forces the network to learn a compressed, lower-dimensional representation of the baseline data distribution.

* **Normal Profiling:** The system trains *exclusively* on standard baseline operational telemetry to map standard feature correlations.
* **Reconstruction Error:** When processing out-of-distribution anomalies, the network struggles to reconstruct the foreign parameters accurately, causing a spike in the Mean Absolute Error (MAE).
* **Statistical Failsafe:** The detection boundary is dynamically computed using a standard statistical variation threshold ( μ + 2σ ), accurately flagging suspicious anomalies.

## 📁 Repository Structure

```text
deep-learning-anomaly-detection/
│
├── config.py          # Centralized network parameters and hyperparameters
├── data_loader.py     # Data synthesis pipeline, transformation, and scaling steps
├── model.py           # Multi-layered Encoder/Decoder Autoencoder layout
├── train.py           # Main training execution pipeline and validation script
└── requirements.txt   # Complete environment dependencies layout
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd deep-learning-anomaly-detection
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Execution

To run the complete data loading, model training, threshold evaluation, and out-of-distribution anomaly inference pipeline, execute the main driver script:

```bash
python train.py
```

## 📊 Core Performance Metrics

* **Bottleneck Latent Representation:** Compresses 20 high-dimensional inputs down to a 6-parameter feature maps core to optimize structural performance.
* **Loss Optimization:** Utilizes Mean Absolute Error (MAE) loss coupled with the Adam optimizer to establish robust reconstruction baselines.
* **Target Detection Efficiency:** Consistently flags >90% of anomalous mutations under test simulation workloads.
