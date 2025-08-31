# Leveraging Persistent Homology Features for Accurate Defect Formation Energy Predictions via Graph Neural Networks

## Description

This work identifies persistent homology features as key indicators for encoding defect information in materials, and can improve the prediction accuracy of state-of-the-art GNN models universally.

## Dependencies

All required packages are listed in `requirements.txt`. Some key dependencies include:

- `python==3.11`
- `torch==2.3.1`
- `torch_geometric==2.5.3`

You can install them with:

```bash
pip install -r requirements.txt
```

## Usage

For graph neural network training and testing, navigate to the `GNN/` directory, and run the script:
```bash
python GNN.py
```

## Citation

If you find this work useful, please consider cite the following reference:

@article{doi:10.1021/acs.chemmater.4c03028,
author = {Fang, Zhenyao and Yan, Qimin},
title = {Leveraging Persistent Homology Features for Accurate Defect Formation Energy Predictions via Graph Neural Networks},
journal = {Chemistry of Materials},
volume = {37},
number = {4},
pages = {1531-1540},
year = {2025},
doi = {10.1021/acs.chemmater.4c03028},
}

}
