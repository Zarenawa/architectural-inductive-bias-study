# Dataset

This project uses a flower image classification dataset containing five categories of flowers. The dataset is not included in this repository and should be downloaded separately.

After obtaining the dataset, place it inside the `datasets/` directory using the following structure:

datasets/
└── flowers/
├── train/
├── val/
└── test/


Each split (`train`, `val`, and `test`) should contain subdirectories corresponding to the flower classes. For example:

datasets/
└── flowers/
├── train/
│ ├── daisy/
│ ├── dandelion/
│ ├── rose/
│ ├── sunflower/
│ └── tulip/
│
├── val/
│ ├── daisy/
│ ├── dandelion/
│ ├── rose/
│ ├── sunflower/
│ └── tulip/
│
└── test/
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/

The dataset location is automatically handled through the configuration files in the `configs/` directory. After placing the dataset in the correct structure, no additional code modification is required.