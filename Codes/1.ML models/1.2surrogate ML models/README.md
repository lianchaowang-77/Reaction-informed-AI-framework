# DFT-Property Surrogate Models

This folder contains eight independently trained DNN surrogate models for predicting DFT-derived molecular properties from SMILES: `Cacetal-q(N)`, `Cacetal-q(N+1)`, `Cacetal-s-`, `E_HOMO(N-1)`, `VIP`, `Overall_Average`, `Pos_Average`, and `Polar_Area`.

Install the required packages with:

```powershell
python -m pip install -r requirements.txt
```

Predict all eight properties for a CSV file containing a SMILES column:

```powershell
python ".\predict_all.py" --input_csv "path\to\molecules.csv" --output_csv "properties.csv"
```

Retrain all eight models with:

```powershell
python ".\train_all.py" --dataset_csv "path\to\dataset.csv" --output_dir ".\trained_models"
```

The `train.py` and `predict.py` files in each target folder can be used to run one property model separately.
