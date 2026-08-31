# Machine-Learning Models

This section contains the models used for hydrolysis-rate prediction and rapid estimation of DFT-derived molecular properties.

## Contents

- **1.1 RD-driven ML models**: eight models that predict `log(k_H)` from a 51-dimensional reaction-descriptor matrix.
- **1.2 Surrogate ML models**: eight single-task DNN models that predict selected DFT-derived properties from molecular SMILES.

## Quick use

Run a hydrolysis-rate prediction with:

```powershell
python ".\1.1RD_driven ML models\DNN\predict.py" --feature_csv "path\to\51D_features.csv" --output_csv "predictions.csv"
```

Predict all eight DFT-derived properties with:

```powershell
python ".\1.2surrogate ML models\predict_all.py" --input_csv "path\to\molecules.csv" --output_csv "properties.csv"
```

More details are provided in the README of each subfolder.
