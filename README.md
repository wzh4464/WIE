# WIE (Window-level Influence Estimator)

This repository implements **WIE (Window-level Influence Estimator)**, a research project that studies SGD influence dynamics in neural networks. The codebase focuses on training neural networks, computing influence functions, and analyzing data cleansing effects across different datasets and model architectures.

Originally containing a BERT sentiment classification pipeline, this repository has evolved into a comprehensive influence function research framework.

## Features

- **Multiple Influence Function Methods**: WIE, LiSSA, DVE, LOO, LAVA, TracIn, TD-Influence and more
- **Epoch-wise Analysis**: Study influence dynamics across training epochs
- **Data Cleansing Experiments**: Identify and remove harmful/mislabeled samples
- **Precision/Recall Metrics**: Evaluate cleansing effectiveness on flip point identification
- **Multiple Datasets**: MNIST, CIFAR-10, EMNIST, Adult Census, 20 Newsgroups
- **Various Model Architectures**: LogReg, DNN, CNN, ResNet, ViT, TinyViT, MobileNetV2
- **Reproducible Results**: Comprehensive seed management and deterministic operations
- **GPU Memory Efficient**: Automatic memory management and batch processing

## Installation

### Requirements

```bash
# Install dependencies using Pixi (conda-based package manager)
pixi install

# Activate the environment
pixi shell

# Run tests
pixi run test
```

## Quick Start

### Basic Training and Influence Computation

Train a model and compute WIE influence scores:

```bash
# Basic training run
python -m wie.training.train --target mnist --model dnn --seed 42 --save_dir results

# Training with data cleansing/relabeling (10% mislabeled samples)
python -m wie.training.train --target mnist --model dnn --seed 42 --save_dir results --relabel 10

# Compute WIE influence functions for all epochs
python -m wie.infl --target mnist --model dnn --seed 42 --type wie_all_epochs --save_dir results
```

### Data Cleansing Experiments

Run influence-based data cleansing experiments:

```bash
# Full cleansing experiment (both precision computation and retraining)
python -m wie.training.exp_influence_cleansing --target mnist --model dnn --seed 42 --type wie_all_epochs --keep_ratio 90 --relabel 10

# Only compute precision/recall/F1 metrics (no retraining)
python -m wie.training.exp_influence_cleansing --target mnist --model dnn --seed 42 --type wie_all_epochs --keep_ratio 90 --relabel 10 --compute_precision True --compute_retraining_loss False

# Only perform retraining on cleansed data (no precision metrics)
python -m wie.training.exp_influence_cleansing --target mnist --model dnn --seed 42 --type wie_all_epochs --keep_ratio 90 --relabel 10 --compute_precision False --compute_retraining_loss True

# Skip both precision and retraining (only sample selection)
python -m wie.training.exp_influence_cleansing --target mnist --model dnn --seed 42 --type wie_all_epochs --keep_ratio 90 --relabel 10 --compute_precision False --compute_retraining_loss False
```

### Epoch-wise Keep Ratio Pipeline

Use the orchestration script for full pipeline execution:

```bash
# Full pipeline: training + influence computation + cleansing
python scripts/epoch_wise_keep_ratio.py --target mnist --model dnn --save_dir results --relabel 10 --seed 42 --keep_ratio 90

# Only compute precision metrics in cleansing step
python scripts/epoch_wise_keep_ratio.py --target mnist --model dnn --save_dir results --relabel 10 --seed 42 --keep_ratio 90 --compute_precision True --compute_retraining_loss False

# Only perform retraining in cleansing step
python scripts/epoch_wise_keep_ratio.py --target mnist --model dnn --save_dir results --relabel 10 --seed 42 --keep_ratio 90 --compute_precision False --compute_retraining_loss True

# Skip training and use existing data
python scripts/epoch_wise_keep_ratio.py --target mnist --model dnn --save_dir new_experiment --relabel 10 --seed 42 --keep_ratio 90 --skip_train --existing_train_dir /path/to/existing/training/data
```

### Cleansing Functionality Control

The cleansing experiments can be controlled with two key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--compute_precision` | `True` | Compute precision/recall/F1 statistics for flip point identification |
| `--compute_retraining_loss` | `True` | Perform actual retraining and compute validation/training losses |

**Use Cases:**

- **Full Analysis** (default): Both parameters `True` - compute precision metrics AND retrain models
- **Precision-Only**: `--compute_precision True --compute_retraining_loss False` - only evaluate cleansing effectiveness, no retraining (faster)
- **Retraining-Only**: `--compute_precision False --compute_retraining_loss True` - only retrain on cleansed data, no precision metrics
- **Sample Selection**: `--compute_precision False --compute_retraining_loss False` - only perform influence-based sample selection

### Supported Influence Methods

| Method | Description | Parameter |
|--------|-------------|-----------|
| `wie_all_epochs` | Window-level Influence Estimator (all epochs) | `--type wie_all_epochs` |
| `icml_all_epochs` | ICML method (all epochs) | `--type icml_all_epochs` |
| `lava_all_epochs` | LAVA method (all epochs) | `--type lava_all_epochs` |
| `dve_all_epochs` | DVE method (all epochs) | `--type dve_all_epochs` |
| `sgd` | SGD-based influence | `--type sgd` |
| `tracin` | TracIn influence | `--type tracin` |
| `td_influence` | TD-Influence | `--type td_influence` |

### Available Datasets and Models

**Datasets:** `mnist`, `cifar10`, `emnist`, `adult`, `20news`
**Models:** `logreg`, `dnn`, `cnn`, `resnet18`, `resnet56`, `vit`, `tinyvit`, `mobilenetv2`

## Project Structure

```
.
├── experiment/            # Core WIE implementation
│   ├── train.py          # Training orchestrator (TrainManager)
│   ├── infl.py          # Influence function computation
│   ├── exp_influence_cleansing.py  # Data cleansing experiments
│   ├── DataModule.py    # Dataset handling (MNIST, CIFAR-10, etc.)
│   ├── NetworkModule.py # Model architectures (DNN, CNN, ResNet, ViT)
│   ├── config.py        # Hyperparameter configurations
│   └── utils.py         # Logging and utility functions
├── scripts/             # Orchestration and batch scripts
│   ├── epoch_wise_keep_ratio.py    # Full pipeline orchestration
│   ├── epoch_wise_cleansing.sh     # Bash experiment runner
│   └── run_epoch_wise_cleansing_batch.sh  # Batch experiments
├── src/                # Bridge to legacy APIs
│   └── experiment/     # Re-exports for compatibility
├── logs/               # Training and experiment logs
├── outputs/            # Experiment outputs and results
└── tests/              # Unit tests
```

### Core Components

- **TrainManager** (`experiment/train.py`): Handles model training, SGD step recording, and checkpointing
- **Influence Computation** (`experiment/infl.py`): Multiple influence function algorithms (WIE, LiSSA, DVE, etc.)
- **Data Cleansing** (`experiment/exp_influence_cleansing.py`): Identifies harmful samples and retrains models
- **Dataset Registry** (`experiment/DataModule.py`): Automatic preprocessing and caching with file locks
- **Model Registry** (`experiment/NetworkModule.py`): Various neural network architectures

### Output Structure

Experiments produce structured outputs:

```
outputs/experiment_name/
├── records/              # Training checkpoints and step data
├── global_info_*.json   # Experiment metadata
├── infl_*.csv          # Influence scores
├── relabel_overlap_*.csv     # Precision/recall metrics
├── cleansed_*_performance_*.csv  # Retraining results
└── kept_indices_*.csv   # Sample selection records
```

## Advanced Usage

### Batch Experiments

Run experiments across multiple seeds:

```bash
# Run cleansing with multiple seeds
python -m wie.training.exp_influence_cleansing --target mnist --model dnn --seed "0,1,2,3,4" --type wie_all_epochs --keep_ratio 90 --relabel 10

# Batch experiment script
bash scripts/run_epoch_wise_cleansing_batch.sh
```

### Custom Hyperparameters

Override training parameters:

```bash
# Custom learning rate and regularization
python -m wie.training.train --target mnist --model dnn --lr 0.001 --num_epoch 10

# Enable learning rate decay
python scripts/epoch_wise_keep_ratio.py --target mnist --model dnn --save_dir results --decay True --lr 0.001
```

### GPU and Memory Management

```bash
# Use specific GPU
python -m wie.training.train --target cifar10 --model resnet18 --gpu 1

# Adjust batch size for memory constraints
python scripts/epoch_wise_keep_ratio.py --target cifar10 --model resnet18 --batch_size 32
```

### Different Influence Types

```bash
# Use DVE influence with projection
python scripts/epoch_wise_keep_ratio.py --target mnist --model dnn --type dve_all_epochs --proj_dim 100

# Use TD-Influence with random projection
python scripts/epoch_wise_keep_ratio.py --target adult --model logreg --type td_influence --use_projection --proj_dim 50
```

## Output Files Explanation

The experiments generate several important output files:

### 1. `relabel_overlap_{seed}.csv`
- **Purpose**: Precision/recall analysis of flip point identification
- **Key Columns**:
  - `precision`: Proportion of dropped samples that were actually mislabeled
  - `recall`: Proportion of mislabeled samples successfully identified
  - `f1_score`: Harmonic mean of precision and recall

### 2. `cleansed_{method}_{keep_ratio}_performance_{seed}.csv`
- **Purpose**: Model performance after retraining on cleansed data
- **Key Columns**:
  - `test_accuracy`: Model accuracy on test set
  - `val_loss`: Validation loss
  - `train_loss`: Training loss on cleansed subset

### 3. `kept_indices_{seed}.csv`
- **Purpose**: Record of which samples were retained each epoch
- **Key Columns**:
  - `kept_indices_preview`: Sample indices kept for training (first 20 shown)

## Precision Analysis Tools

Analyze and compare precision performance across different influence methods and configurations:

### Analyze Existing Results

```bash
# Basic precision analysis
python scripts/analyze_precision_performance.py --base_dir outputs/sentiment_experiments

# With visualizations
python scripts/analyze_precision_performance.py --base_dir outputs --target sentiment --model bert --plot

# Using different metric for best configuration
python scripts/analyze_precision_performance.py --base_dir outputs --metric f1_max --output_dir results

# Bash script wrapper
bash scripts/run_precision_analysis.sh -d outputs/sentiment_experiments -p
```

### Generate Test Data and Run Example

```bash
# Create test data and run analysis example
python scripts/example_precision_analysis.py

# Only create test data
python scripts/example_precision_analysis.py --create-data

# Only run analysis on existing test data
python scripts/example_precision_analysis.py --run-analysis
```

### Analysis Features

- **Multi-method Comparison**: Compare WIE, ICML, LAVA, and DVE methods
- **Configuration Optimization**: Find best keep_ratio and relabel_percentage combinations
- **Statistical Summary**: Mean, max, and final epoch metrics across seeds
- **Visualization Support**: Generate plots and heatmaps (requires matplotlib/seaborn)
- **Automated Reports**: Generate comprehensive text reports with rankings

## Grid Search Experiments

Run large-scale parameter sweeps across multiple influence methods and configurations:

### Quick Start Grid Search

```bash
# Basic grid search with default parameters
bash scripts/run_cleansing_grid.sh

# Dry run to see what would be executed
bash scripts/run_cleansing_grid.sh --dry-run

# Small configuration for testing
bash scripts/run_cleansing_grid.sh -g configs/sentiment_cleansing_small.json --dry-run

# Run with automatic precision analysis
bash scripts/run_cleansing_grid.sh --run-analysis
```

### Custom Grid Configurations

```bash
# Create custom grid configuration
python scripts/create_grid_config.py --preset small -o configs/my_grid.json

# Use custom methods and parameters
python scripts/create_grid_config.py \
  --methods wie_all_epochs icml_all_epochs \
  --keep-ratios 70 80 90 \
  --relabel-percentages 10 20 30 \
  --seeds 0 1 2 3 \
  -o configs/custom_grid.json

# Run with custom configuration
bash scripts/run_cleansing_grid.sh -g configs/custom_grid.json --run-analysis
```

### Advanced Grid Options

```bash
# Precision-only mode (faster, no retraining)
bash scripts/run_cleansing_grid.sh --compute-retraining False --run-analysis

# Custom GPU allocation
bash scripts/run_cleansing_grid.sh --gpus 0,1,2,3

# Different target and model
bash scripts/run_cleansing_grid.sh -t cifar10 -m resnet18

# Direct Python interface
python scripts/run_influence_cleansing_grid.py \
  --grid-file configs/influence_cleansing_grid.json \
  --output-root outputs \
  --gpus 0 1 2 \
  --run-analysis \
  --dry-run
```

### Grid Configuration Presets

| Preset | Description | Methods | Experiments |
|--------|-------------|---------|-------------|
| `full` | Complete comparison | WIE, ICML, LAVA, DVE | 192 experiments |
| `small` | Quick test | WIE, ICML | 8 experiments |
| `wie-only` | Focus on WIE | WIE only | 100 experiments |
| `fast` | Minimal test | WIE, ICML | 4 experiments |
| `precision-only` | Extended precision analysis | All methods | 384 experiments |

### Grid Search Features

- **Multi-GPU Distribution**: Automatically distributes experiments across available GPUs
- **Fault Tolerance**: Individual experiment failures don't stop the entire grid
- **Progress Tracking**: Real-time progress reporting for each GPU worker
- **Automatic Analysis**: Optional integrated precision analysis after completion
- **Flexible Configuration**: JSON-based grid specification with CLI overrides
- **Dry Run Mode**: Preview all commands before execution

## Research Background

WIE (Window-level Influence Estimator) studies how the influence of training samples changes throughout the SGD training process. This is crucial for understanding:

- **Data Quality**: Identifying harmful or mislabeled samples
- **Training Dynamics**: How sample importance evolves over epochs
- **Model Robustness**: Effects of removing problematic training data

## Key Publications

This implementation supports research on influence functions and data cleansing, including methods from:
- WIE (Window-level Influence Estimator)
- TracIn (Tracing Training Data Influence)
- DVE (Data Valuation using Reinforcement Learning)
- LAVA (Learning to Remove Data Efficiently)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size for large models:

```bash
python scripts/epoch_wise_keep_ratio.py --target cifar10 --model resnet18 --batch_size 16
```

### Slow Experiments

Use precision-only mode to avoid retraining:

```bash
python -m wie.training.exp_influence_cleansing --target mnist --model dnn --compute_retraining_loss False
```

### Reproducibility

All experiments use comprehensive seed management for full reproducibility:

```bash
python -m wie.training.train --target mnist --model dnn --seed 42
```
