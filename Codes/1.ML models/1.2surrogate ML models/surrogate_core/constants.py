from __future__ import annotations


TARGETS = (
    "Cacetal-q(N)",
    "Cacetal-q(N+1)",
    "Cacetal-s-",
    "E_HOMO(N-1)",
    "VIP",
    "Overall_Average",
    "Pos_Average",
    "Polar_Area",
)

FINGERPRINT = {
    "type": "hashed_morgan_counts",
    "radius": 2,
    "n_bits": 1024,
    "use_chirality": False,
}

MODEL_SETTINGS = {
    "name": "DNN22_tuned",
    "hidden_units": [512, 256, 128],
    "dropout": 0.15,
    "l2": 0.0,
    "learning_rate": 0.001,
    "loss": "mean squared error",
    "epochs_max": 300,
    "batch_size": 64,
    "early_stopping_patience": 25,
    "reduce_lr_patience": 10,
    "reduce_lr_factor": 0.5,
}
