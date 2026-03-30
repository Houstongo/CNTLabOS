$ErrorActionPreference = "Stop"

$projectRoot = "D:\CNTDATA\CNTA_ML_Project"
$pythonExe = "C:\Users\clearlove\.conda\envs\lab_agent\python.exe"
$trainScript = Join-Path $projectRoot "experiments\cnt_paper_repro\train.py"
$configPath = Join-Path $projectRoot "experiments\cnt_paper_repro\configs\paper_100000x_cldice_taskadapted_gpu_rerun.yaml"
$logRoot = Join-Path $projectRoot "experiments\cnt_paper_repro\runs\launch_logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$queueLog = Join-Path $logRoot "taskadapted_expc_gpu_rerun_$timestamp.log"
$stdoutLog = Join-Path $logRoot "taskadapted_expc_gpu_rerun_$timestamp.stdout.log"
$stderrLog = Join-Path $logRoot "taskadapted_expc_gpu_rerun_$timestamp.stderr.log"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $queueLog -Append
}

Set-Location $projectRoot
Write-RunLog "START config=$configPath"

$commandLine = "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && `"$pythonExe`" `"$trainScript`" --config `"$configPath`" 1>`"$stdoutLog`" 2>`"$stderrLog`""
& cmd.exe /d /c $commandLine

if ($LASTEXITCODE -ne 0) {
    Write-RunLog "FAILED exit_code=$LASTEXITCODE stdout_log=$stdoutLog stderr_log=$stderrLog"
    exit $LASTEXITCODE
}

Write-RunLog "DONE stdout_log=$stdoutLog stderr_log=$stderrLog"
