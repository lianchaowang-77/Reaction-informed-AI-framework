$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "E:\software\miniconda\anzhuang\envs\lumia\python.exe"
$Runner = Join-Path $Root "run_5methods_with_run_convergence.py"

& $Python $Runner
if ($LASTEXITCODE -ne 0) {
    throw "Five-method targeted search failed with exit code $LASTEXITCODE"
}
