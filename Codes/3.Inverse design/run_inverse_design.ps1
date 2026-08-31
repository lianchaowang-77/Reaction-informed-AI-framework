param(
    [string]$PythonExe = "E:\software\miniconda\anzhuang\envs\lumia\python.exe",
    [string]$PackDir = "E:\2025\ML\permeability\PolymerGasMembraneML\datasets\acetal\pack_uncertainty_5seed",
    [double[]]$Targets = @(6.099),
    [int]$EvalsPerRun = 20000,
    [int]$Seed = 1,
    [string]$Device = "cpu",
    [int]$BatchSize = 4096,
    [double]$StrictRelErr = 0.01,
    [double]$RelaxedRelErr = 0.05,
    [int]$StrictRuns = 10,
    [int]$MaxRuns = 15,
    [int]$ConvergeTopN = 5,
    [int]$HistoryKeep = 20,
    [string]$OutRoot = ""
)

$ErrorActionPreference = "Stop"
$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PoolCsv = Join-Path $BaseDir "pool_union_for_inverse.csv"

if (-not $OutRoot -or $OutRoot.Trim() -eq "") {
    $OutRoot = Join-Path $BaseDir "results"
}

$Methods = @(
    @{ Name = "BO";    Script = "bo.py";    Extra = @("--n_init", "300", "--n_candidates", "10000", "--propose_batch", "500") },
    @{ Name = "GA";    Script = "ga.py";    Extra = @() },
    @{ Name = "NSGA2"; Script = "nsga2.py"; Extra = @() },
    @{ Name = "MCTS";  Script = "mcts.py";  Extra = @("--sim_batch", "128") },
    @{ Name = "RL";    Script = "rl.py";    Extra = @("--sample_batch", "256") }
)

foreach ($Target in $Targets) {
    $TargetText = "{0:0.###}" -f $Target
    $TargetDir = Join-Path $OutRoot ("target_" + $TargetText)

    foreach ($Method in $Methods) {
        $ScriptPath = Join-Path $BaseDir $Method.Script
        $OutDir = Join-Path $TargetDir $Method.Name
        New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

        $CommonArgs = @(
            $ScriptPath,
            "--pack_dir", $PackDir,
            "--out_dir", $OutDir,
            "--target", $TargetText,
            "--runs", $MaxRuns,
            "--evals_per_run", $EvalsPerRun,
            "--seed", $Seed,
            "--device", $Device,
            "--batch_size", $BatchSize,
            "--pool_union_csv", $PoolCsv,
            "--adaptive_convergence",
            "--strict_rel_err", $StrictRelErr,
            "--relaxed_rel_err", $RelaxedRelErr,
            "--strict_runs", $StrictRuns,
            "--max_runs", $MaxRuns,
            "--converge_top_n", $ConvergeTopN,
            "--history_keep", $HistoryKeep
        )

        $AllArgs = $CommonArgs + $Method.Extra
        Write-Host "[RUN] target=$TargetText method=$($Method.Name)"
        & $PythonExe @AllArgs
        if ($LASTEXITCODE -ne 0) {
            throw "[$($Method.Name)] target=$TargetText failed with exit code $LASTEXITCODE"
        }
    }
}

Write-Host "[OK] Inverse-design jobs finished. Output root: $OutRoot"
