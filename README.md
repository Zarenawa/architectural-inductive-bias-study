# Architectural Inductive Bias in Image Classification

This repository contains the implementation and experimental results for the study:

**"Architectural Inductive Bias in Image Classification: A Comparative Study of MLP, CNN, and RNN"**

This work investigates the influence of architectural inductive bias on image classification performance by comparing three neural network architectures: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN). All models are trained and evaluated under the same experimental protocol using a flower image classification dataset.

The objective of this study is to analyze how different architectural structures affect the ability of neural networks to learn visual representations. In particular, the study examines whether architectures designed with image-specific spatial inductive bias provide advantages over architectures without explicit spatial modeling.

## Models Compared

The following architectures are evaluated:

- **Multi-Layer Perceptron (MLP):** A fully connected network that processes flattened image features without explicitly preserving spatial relationships.
- **Convolutional Neural Network (CNN):** A convolution-based architecture that exploits local receptive fields and weight sharing to capture spatial patterns in images.
- **Recurrent Neural Network (RNN):** A sequential architecture that processes image information as ordered sequences.

## Experimental Results

The quantitative evaluation demonstrates that the CNN achieves the highest classification performance among the three architectures.

 | Macro F1-score | Macro AUC |

Model  Top-1Acc  MacroPre  MacroRec   F1score   Auc
| MLP | 44.09%  | 45.31%  | 43.88%  | 44.09%  | 77.63% |
| CNN | 68.67%  | 68.84%  | 68.32%  | 67.98%  | 91.28% |
| RNN | 45.06%  | 44.93%  | 44.90%  | 43.88%  | 77.86% |

The results indicate that incorporating image-specific spatial inductive bias provides a significant advantage for image classification tasks.

## Repository Structure

architectural-inductive-bias-study/

├── configs/
│ ├── flower_cnn.py
│ ├── flower_mlp.py
│ └── flower_rnn.py
│
├── core_modules/
│ ├── cnn.py
│ ├── mlp.py
│ ├── rnn.py
│ ├── dataloader.py
│ ├── model.py
│ └── init.py
│
├── datasets/
│ └── README.md
│
├── experiments/
│ └── flower/
│ └── cnn/
│ └── notebooks/
│
├── paper/
│ └── research paper PDF
│
└── README.md


## Installation

The experiments were conducted using:

- Python 3.10.20
- PyTorch
- NVIDIA CUDA 13.0

Create a Python environment and install the required dependencies:

conda activate gpu


Dataset Preparation
The dataset is not included in this repository.
Please refer to: "datasets/README.md" for instructions on downloading and organizing the dataset.

Running Experiments
All experiments are controlled through configuration files located in: "configs/" To select an experiment, modify the configuration module in the training notebook: 
CONFIG_MODULE = 'configs.flower_cnn'

The configuration file controls the dataset paths, model selection, training parameters, optimization settings, and experiment output locations.

After selecting the desired configuration, run: 
experiments/flower/cnn/notebooks/train_model.ipynb

The framework automatically creates the required experiment folders and organizes:

trained model weights
training plots
evaluation results
training summaries

No manual folder creation is required.

## Extending the Framework

The project is designed with a modular structure to allow additional models and datasets.

Adding a New Model
To add a new architecture:

1. Implement the model in: core_modules/
2. Register the model in: core_modules/model.py
3. create a corresponding config file in: configs/
4. select the new config through: CONFIG_MODULE = 'configs.new_model'

## Adding a New Dataset
New datasets can be added by creating a new experiment directory: 

experiments/
└── dataset_name/

A corresponding configuration file should then define the dataset paths, number of classes, input size, and training settings.
The existing training pipeline can be reused without modifying the core training logic.

## Paper
The complete paper is available in the paper/ directory.