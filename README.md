# MobileNetV4-PyTorch

## Overview
This repository provides a PyTorch replication of the MobileNetV4 architecture as described in the paper ["MobileNetV4: Universal Models for the Mobile Ecosystem"](https://arxiv.org/pdf/2404.10518v1). The implementation aims to mimic the architecture closely for all five variants: MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge, MobileNetV4HybridMedium, and MobileNetV4HybridLarge.

## Repository Structure
```
.
├── env
│   └── MobileNetV4_env.yml
├── logs
│   ├── MobileNetV4ConvLarge_architecture.txt
│   ├── MobileNetV4ConvMedium_architecture.txt
│   ├── MobileNetV4ConvSmall_architecture.txt
│   ├── MobileNetV4HybridLarge_architecture.txt
│   └── MobileNetV4HybridMedium_architecture.txt
├── MobileNetV4.py
├── nn_blocks.py
├── paper
│   └── 2404.10518v1.pdf
└── test.py
```
- **env/**: Contains the environment YAML file to set up the necessary dependencies.
- **logs/**: Contains the architecture details of the different MobileNetV4 variants.
- **paper/**: Contains the original MobileNetV4 paper for reference.
- **MobileNetV4.py**: Contains the feature extractor for MobileNetV4 architectures.
- **nn_blocks.py**: Contains neural network block definitions used in the MobileNetV4 architecture.
- **test.py**: Contains the classifier and script for testing the implementations.

## Installation
To create the environment with the necessary dependencies, use the provided YAML file:

```bash
conda env create -f env/MobileNetV4_env.yml
conda activate MobileNetV4-PyTorch
```

## Usage
### Training
To train a MobileNetV4 model on your dataset, modify the `test.py` script with your dataset and training configurations.

### Pre-trained Weights
For pre-trained weights on ImageNet, you can use the weights provided by [timm](https://huggingface.co/collections/timm/mobilenetv4-pretrained-weights-6669c22cda4db4244def9637).

### Example
```python
import torch
from test import MobileNetV4WithClassifier

# Example usage
model = MobileNetV4WithClassifier(model_name='MobileNetV4ConvSmall', num_classes=1000)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output)
```

## Citations
If you find this work useful, please cite the original MobileNetV4 paper:
```bibtex
@article{MobileNetV4,
  title={MobileNetV4: Universal Models for the Mobile Ecosystem},
  author={Author Names},
  journal={arXiv preprint arXiv:2404.10518v1},
  year={2024}
}
```
If you use this work, please cite it as follows:
```bibtex
@misc{MobileNetV4-PyTorch,
  author = {Muhammad Junaid Ali Asif Raja},
  title = {MobileNetV4-PyTorch},
  year = {2024},
  url = {https://github.com/junaidaliop/MobileNetV4},
  note = {Version 1.0.0}
}
```

## Contact
For research collaborations or any queries, please email me at [muhammadjunaidaliasifraja@gmail.com](mailto:muhammadjunaidaliasifraja@gmail.com)

## Contributions
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## TODO
- [ ] Train the model on ImageNet to attain weights
- [ ] Train the model on CIFAR-100
- [ ] Train the model on CIFAR-10

## Star the Repository
If you find this repository useful, please consider giving it a star!

## Acknowledgment

This is an unofficial implementation of MobileNetV4 in PyTorch. To the best of my ability, I believe this is the closest implementation to the original work found at [TensorFlow MobileNetV4 Implementation](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py).

For the official TensorFlow implementation, please visit: [TensorFlow MobileNetV4 Implementation](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py).
