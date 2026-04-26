# RF Modulation Classification with Brain-Inspired Models

A unified experiment framework for comparing spiking and non-spiking neural architectures on RF modulation classification — investigating where neuromorphic approaches offer genuine advantages over standard deep learning, and where they don't.

Built on the RadioML 2018.01A dataset (24 modulation classes, 26 SNRs, 20GB), this framework supports systematic comparison across architectures, encoders, and training rules from a single YAML config. Developed during graduate studies in Neuromorphic Computing (D7064E) at Luleå University of Technology, 2025.
 
## Research Questions
 
- To what extent can SNNs achieve competitive performance on RF modulation classification compared to ANN baselines?
- How do different RF-to-spike encoding strategies influence classification accuracy and learning stability?
- How does training method (BPTT vs DCLL vs local rules) affect performance, convergence, and decision latency?
- How robust are SNN classifiers to varying SNR compared to ANN baselines — and where does the gap widen or close?
- What trade-offs exist between accuracy, temporal dynamics, spike activity, and hardware feasibility across architectures?

## Key Results
 
### Classification accuracy (6-class subset, Conv2D-SNN, IQ Grid encoding)
 
| SNR | Accuracy |
|-----|----------|
| −20 dB | ~19% |
| −10 dB | ~23% |
| 0 dB | ~31% |
| 10 dB | ~80% |
| 20 dB | ~85% |
| 30 dB | **87%** |
 
**Full 24-class results:** 56% accuracy at SNR 30 dB — performance degrades predictably with class count and SNR.
 
### Architecture comparison (6-class, SNR 30 dB)
 
| Architecture | Notes |
|-------------|-------|
| **Conv2D-SNN** | Best overall performer |
| ANN baseline | Comparable accuracy, 1.8x training time |
| HRM (hierarchical recurrent) | High accuracy, higher complexity — promising |
| LSM / Reservoir | Fast to train, lower accuracy — needs further work |
 
### Encoding comparison
 
| Encoding | Result |
|---------|--------|
| **IQ Grid** | Best accuracy overall |
| **Sigma-delta** | Best noise robustness — strongest under packet dropout and out-of-order data |
| TF Events | Similar to Latency TTFS |
| Latency TTFS | Similar to TF Events |
| Level Crossing ADC | Lowest accuracy |
| Rate Encoding | Weakest performer |
 
### Training rule comparison
 
BPTT and DCLL achieved very similar performance. Local rule produced slightly lower accuracy. DCLL's time-distributed supervision showed advantages for temporal robustness in some configurations.
 
## Key Observation on SNR Robustness
 
SNR 10, 20, and 30 dB produced similar performance — suggesting that for practical deployment, training and evaluation could focus on this range to reduce compute cost significantly. Full −20 to +30 dB sweeps required ~18 hour training runs (subsampled dataset, CPU).
 
## Framework Design
 
The framework separates five concerns cleanly — dataset, encoding, model, trainer, and logging — each controlled via a single YAML config. This makes it straightforward to run systematic sweeps and track results across runs without modifying code.
 
```yaml
experiment:  # name, seed, output directory
dataset:     # RadioML subset — modulations, SNRs, train/val/test split
encoding:    # RF → spikes/events transform
model:       # neural architecture and hyperparameters
training:    # optimizer, trainer type, epochs, device
logging:     # per-SNR metrics, spike stats, decision latency
```
 
Each run writes results to a timestamped subdirectory:
```
results/<experiment_name>_<YYYYMMDD_HHMMSS>/
    config.yaml
    metrics.json
    metrics_per_epoch.json
    train.log
```
 
## Supported Architectures
 
| Model | Type | Input Shape | Notes |
|-------|------|------------|-------|
| Conv2D-SNN | Spiking CNN | `[B, T, C, H, W]` | LIF neurons, IQ grid input |
| Conv1D-SNN | Spiking CNN | `[B, T, C]` | 1D sequences, rate/sigma-delta input |
| Conv2D-ANN | Non-spiking baseline | `[B, T, C, H, W]` | ReLU, matched architecture for fair comparison |
| LSM / ESN | Reservoir | `[B, T, F]` | Fixed random reservoir, fast training |
| HRM-RF | Hierarchical recurrent | `[B, T, F]` | Slow/fast GRUs, multi-cycle processing |
 
## Supported Encoders
 
| Encoder | Description | Sparsity | Best For |
|---------|-------------|---------|---------|
| `iq_grid` | IQ samples mapped to 2D energy grid | Low | Conv2D accuracy |
| `rate` | Rate-coded spike probability from I/Q | Medium | Simple baseline |
| `sigma_delta` | Events on IQ value changes | High | Noise robustness |
| `tf_events` | Time-frequency events from short-time transform | Medium | Transient features |
| `latency_ttfs` | Time-to-first-spike — amplitude → spike time | Very high | Low-latency decisions |
| `level_crossing_adc` | Spikes at quantisation level crossings | High | Hardware efficiency |
 
## Supported Training Rules
 
| Trainer | Description | Notes |
|---------|-------------|-------|
| `bptt` | Full backpropagation through time | Highest accuracy, compute heavy |
| `dcll` | Time-distributed loss — supervision at every timestep | Strong temporal robustness |
| `local` | Local-in-time rules | More biologically plausible, lower accuracy |
 
## Installation
 
```bash
conda create -n snn311 python=3.11
conda activate snn311
pip install torch torchvision torchaudio
pip install numpy h5py pyyaml pandas matplotlib
```
 
Download the RadioML 2018.01A dataset (~20GB) from [Kaggle](https://www.kaggle.com/datasets/pinxau1000/radioml2018/data) and place it at:
```
data/GOLD_XYZ_OSC.0001_1024.hdf5
```
 
## Running Experiments
 
```bash
# Single experiment from config
python run_experiment.py --config configs/radioml2018_full_hrm_bptt.yaml
 
# Aggregate and analyse all results
python analyze_experiments.py --root results --out_csv all_runs_summary.csv
 
# Filter by class count
python analyze_experiments.py --root results --num-classes 2 24
```
 
### Quick debug run (2-class, limited SNR, CPU)
 
```yaml
dataset:
  frames_per_combo: 256
  mods_to_use: [3, 4]
  snrs_to_use: [0, 10]
training:
  batch_size: 16
  epochs: 5
  device: cpu
```
 
## Repository Structure
 
```
.
├── run_experiment.py           # Entry point — trains a single config
├── analyze_experiments.py      # Aggregates and visualises results across runs
├── configs/                    # YAML experiment configurations
└── rf_experiments/
    ├── runner.py               # ExperimentRunner — builds data, model, trainer
    ├── dataset.py              # RadioML2018 dataset wrapper
    ├── encoders/               # Six RF→spike encoding strategies
    ├── models/                 # Five model architectures
    ├── trainers/               # BPTT, DCLL, local rule trainers
    └── analysis.py             # Result collection and plotting utilities
```
 
## Analysis Utilities
 
`rf_experiments/analysis.py` provides:
- `plot_acc_vs_snr()` — accuracy vs SNR for selected runs
- `plot_encoding_model_heatmap()` — test accuracy by encoder/model combination
- `plot_learning_curves_for_run()` — per-epoch train/val loss and accuracy
- `plot_run_summary()` — compact per-run panel with accuracy curves, per-SNR bars, spike statistics, and decision latency histograms
## Lessons Learned
 
- **Compute matters:** Most training runs were on CPU. Full 24-class, 5-SNR runs took ~18 hours. Access to GPU or institutional compute would have enabled significantly more systematic exploration.
- **SNR subset strategy:** SNR 10/20/30 dB produced similar performance — future work could focus on this range to reduce training cost without sacrificing insight.
- **Framework value:** A structured experiment framework with YAML configs and automatic result tracking was essential for systematic comparison. Running 30+ experiments without it would have been unmanageable.
## References
 
Selected key references:
 
- Smith et al. (2024). Effects of RF Signal Eventization Encoding on Device Classification Performance. *Electronics*
- Weathers et al. (2025). An Event-Based Time-Incremented SNN Architecture Supporting Energy-Efficient Device Classification. *Electronics*
- Pritchard et al. (2024). RFI detection with spiking neural networks. *Publications of the Astronomical Society of Australia*
- Stuijt et al. (2021). μBrain: An event-driven and fully synthesizable architecture for spiking neural networks. *Frontiers in Neuroscience*