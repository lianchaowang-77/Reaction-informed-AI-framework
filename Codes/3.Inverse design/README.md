# Inverse Design

This folder contains five algorithms—BO, GA, NSGA-II, RL, and MCTS—for designing molecules whose predicted hydrolysis rates match a specified target. All methods use the same layered substituent-search strategy and convergence criteria.

Run all five methods for a target `log(k_H)` value with:

```powershell
powershell -ExecutionPolicy Bypass -File ".\run_inverse_design.ps1" -PythonExe "path\to\python.exe" -PackDir "path\to\pack_uncertainty_5seed" -Targets 6.099
```

The results are written to the local `results` folder unless another output path is specified.
