## Fork Notice

This repository is a **fork** of the original [`wa-hls4ml`](github.com/Dendendelen/wa-hls4ml).

This fork extends the original work for use in my thesis.

All credit for the original implementation goes to the original authors.

# wa-hls4ml: Surrogate Model for HLS Synthesis

`wa-hls4ml` is a machine learning framework designed to predict the performance metrics (latency and resource usage) of HLS (High-Level Synthesis) designs. It supports both standard Multilayer Perceptrons (MLP) and Graph Neural Networks (GNN) to model the structure of the synthesis tasks.

## Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Dataset Structure](#dataset-structure)
3.  [Usage](#usage)
    *   [Training](#training)
    *   [Testing](#testing)
    *   [Arguments](#available-arguments)
4.  [Models](#models)
5.  [Output](#output)

---

## Prerequisites

To run this tool, you need Python installed with the following libraries:
-   `torch` (PyTorch)
-   `torch_geometric` (for GNN models)
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `plotly`

Install dependencies via pip:
```bash
pip install torch torch-geometric numpy pandas matplotlib plotly
```

---

## Dataset Structure

The tool expects the dataset to be organized into three separate directories for **Training**, **Validation**, and **Testing**. Each directory should contain one or more `.json` files.

Each JSON file contains a list of model synthesis reports. The key fields required in the JSON objects differ slightly but generally include:
-   `meta_data`: Model name and file paths.
-   `model_config`: Configuration of layers (used to build the graph/model string).
-   `hls_config`: Precision and Reuse Factor settings.
-   `latency_report`: `cycles_min`, `cycles_max`, `target_clock`, etc.
-   `resource_report`: `BRAM`, `DSP`, `FF`, `LUT`, `URAM`.

**Directory Layout Example:**
The dataset for wa-hls4ml can be found/downloaded at [HuggingFace](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml).

The dataset layout should look like this:

```text
dataset/
├── train/
│   ├── train_part1.json
│   └── train_part2.json
├── val/
│   └── val_data.json
└── test/
    └── test_data.json
```

---

## Usage

The main entry point is `wa_hls4ml.py`.

### Training

To train a new model, you simply point the tool to your dataset directories. You can choose to train a **Classifier** (predicts synthesis success), a **Regressor** (predicts latency/resources), or both.

#### Example: Training a Regression Model (MLP)
This trains a simple MLP regressor to predict latency and resource usage.
```bash
python3 wa_hls4ml.py --train --regression \
    --train-dir dataset/train \
    --val-dir dataset/val \
    --test-dir dataset/test \
    -f my_model_output
```

#### Example: Training with Graph Neural Networks (GNN)
Add the `--gnn` (or `-g`) flag to use a Graph Neural Network, which explicitly models the connectivity of the HLS design layers.
```bash
python3 wa_hls4ml.py --train --regression --gnn \
    --train-dir dataset/train \
    --val-dir dataset/val \
    --test-dir dataset/test \
    -f my_gnn_model
```

### Testing

To test an existing trained model on a dataset.
```bash
python3 wa_hls4ml.py --test --regression \
    --test-dir dataset/test \
    -f my_model_output
```

### Available Arguments

| Argument | Description |
| :--- | :--- |
| `--train` | Flag to train a new model. |
| `--test` | Flag to test an existing model. |
| `--classification`, `-c` | Train/Test the binary classifier (Synthesis Success: T/F). |
| `--regression`, `-r` | Train/Test the regressor (Latency, LUTs, DSPs, etc.). |
| `--gnn`, `-g` | Use a Graph Neural Network (GNN) instead of a simple MLP. |
| `--train-dir <path>` | Path to the directory containing training JSON files. |
| `--val-dir <path>` | Path to the directory containing validation JSON files. |
| `--test-dir <path>` | Path to the directory containing testing JSON files. |
| `-f <name>`, `--folder` | **Required.** Name of the output folder where models and results will be saved (inside `models/`). |
| `--no-tts` | **Required if using separate dirs.** Disables internal train-test split (automatically set if using dirs). |
| `--gpu` | Use CUDA GPU for training if available. |

---

## Models

### 1. MLP (Default)
A standard Multi-Layer Perceptron.
-   **Features:** Uses scalar features like bit-width (`prec`), Reuse Factor (`rf`), input/output dimensions (`d_in`, `d_out`), and strategy.
-   **Architecture:** Several fully connected layers with ReLU/ELU activations and Dropout.
-   **Best for:** Simple aggregated predictions where graph structure is less critical.

### 2. GNN (Graph Neural Network)
Enabled with `--gnn`.
-   **Features:** detailed graph representation of the HLS model.
    -   **Nodes:** Layers (e.g., Dense, Conv2D), with features like layer size.
    -   **Edges:** Connectivity between layers.
    -   **Globals:** Global configurations like Precision and Reuse Factor.
-   **Architecture:** Uses Graph Attention Networks (GATv2) and Message Passing to aggregate information across the design before global pooling.
-   **Best for:** Complex topologies where the interaction between specific layers impacts performance.

---

## Output

All outputs are saved in `models/<folder_name>/`.

-   **CSVs**: `train_data.csv`, `val_data.csv`, `test_data.csv` (Parsed intermediate data).
-   **Models**: Saved PyTorch state dicts (e.g., `model.pth`, `model_weights.pth`).
-   **Plots**:
    -   Loss curves for training/validation.
    -   Scatter plots comparing predicted vs. actual values (for Test runs).
    -   Histograms of residuals.
