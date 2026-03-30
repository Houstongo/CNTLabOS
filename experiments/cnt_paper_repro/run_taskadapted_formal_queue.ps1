$ErrorActionPreference = "Stop"

$projectRoot = "D:\CNTDATA\CNTA_ML_Project"
$pythonExe = "C:\Users\clearlove\.conda\envs\lab_agent\python.exe"
$trainScript = Join-Path $projectRoot "experiments\cnt_paper_repro\train.py"
$logRoot = Join-Path $projectRoot "experiments\cnt_paper_repro\runs\launch_logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$queueLog = Join-Path $logRoot "taskadapted_formal_queue_$timestamp.log"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $queueLog -Append
}

$configs = @(
    @{
        Name = "taskadapted_baseline"
        Path = Join-Path $projectRoot "experiments\cnt_paper_repro\configs\paper_100000x_taskadapted_gpu.yaml"
    },
    @{
        Name = "taskadapted_expc"
        Path = Join-Path $projectRoot "experiments\cnt_paper_repro\configs\paper_100000x_cldice_taskadapted_gpu.yaml"
    }
)

Set-Location $projectRoot
Write-QueueLog "Queue started."

foreach ($config in $configs) {
    $name = $config.Name
    $configPath = $config.Path
    $stdoutLog = Join-Path $logRoot "$name`_$timestamp.stdout.log"
    $stderrLog = Join-Path $logRoot "$name`_$timestamp.stderr.log"

    Write-QueueLog "START $name config=$configPath"
    $commandLine = "`"$pythonExe`" `"$trainScript`" --config `"$configPath`" 1>`"$stdoutLog`" 2>`"$stderrLog`""
    & cmd.exe /d /c $commandLine
    if ($LASTEXITCODE -ne 0) {
        Write-QueueLog "FAILED $name exit_code=$LASTEXITCODE stdout_log=$stdoutLog stderr_log=$stderrLog"
        exit $LASTEXITCODE
    }
    Write-QueueLog "DONE $name stdout_log=$stdoutLog stderr_log=$stderrLog"
}

Write-QueueLog "QUEUE_DONE"
