# RD-Driven Hydrolysis-Rate Models

This folder contains eight models for predicting `log(k_H)` from the 51-dimensional reaction-descriptor (RD) matrix: DNN, Ridge, Lasso, SVR, LGBM, XGB, RF, and ET. Each model includes five pretrained members (`seed1`–`seed5`) and separate training and prediction scripts.

Install the required packages with:

```powershell
python -m pip install -r requirements.txt
```

Predict `log(k_H)` with a pretrained model, for example DNN:

```powershell
python ".\DNN\predict.py" --feature_csv "path\to\51D_features.csv" --output_csv "predictions.csv"
```

Retrain a model with the experimental dataset and RD matrix:

```powershell
python ".\DNN\train.py" --dataset_csv "path\to\dataset.csv" --feature_csv "path\to\51D_features.csv" --feature_smiles_csv "path\to\feature_smiles.csv" --target_col "LOGk2" --output_dir ".\DNN\trained"
```

Replace `DNN` with another model folder to run the corresponding algorithm.
