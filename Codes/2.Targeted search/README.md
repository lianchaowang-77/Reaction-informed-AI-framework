# Targeted Search

This folder contains five algorithms—BO, GA, NSGA-II, RL, and MCTS—for searching molecules with predicted `log(k_H) > 6.1`. During each search iteration, 30% of the candidates are generated using selected R3–R4 substituent combinations as structural guidance.

Before running, set the `PYTHON` and `PACK` paths at the beginning of `run_5methods_with_run_convergence.py`. Then run all five methods with:

```powershell
powershell -ExecutionPolicy Bypass -File ".\run_targeted_search.ps1"
```

The search results are written to the local `results` folder.
