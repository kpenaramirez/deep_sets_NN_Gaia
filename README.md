# Astrophysical data analysis with DeepSets neural networks on Gaia star cluster catalogues.

This repository contains the Python script **`gaia_oc_amd.py`**, which integrates astrophysical data analysis with machine learning methods.

## Overview
The project leverages the Gaia star catalogues to build structured datasets and applies **DeepSets neural networks** (Zaheer et al.2017) implemented in PyTorch for classification tasks. It’s a practical example of applying **representation learning on sets** to real-world scientific data.

- **Data Engineering**: Querying and preprocessing Gaia catalogues, cones, and isochrones into clean feature sets.
- **Feature Engineering**: Adding astrophysical features and statistical properties for model training.
- **Machine Learning**: Training DeepSets models with PyTorch to predict cluster membership probabilities.
- **Evaluation & Visualization**: Metrics plotting, candidate evaluation, and membership probability visualization.


## Requirements
- Python 3.9+
- Libraries: `numpy`, `pandas`, `matplotlib`, `torch`
- Gaia credentials file for querying catalogues

## Usage
Clone the repository and run:
```bash
python gaia_oc_amd.py

## Why It Matters
This project illustrates how data science workflows — from ETL to feature engineering and deep learning — can be applied to scientific domains outside traditional business analytics, showcasing the versatility of machine learning.


## License
GNU AGPLv3
