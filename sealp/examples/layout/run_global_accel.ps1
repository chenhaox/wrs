param(
    [int]$Explore = 80,
    [int]$MaxEvals = 220,
    [int]$LegacyReplay = 28,
    [int]$SearchGraspCap = 160
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$PY = "D:\Soft\tools\anaconda\envs\spatialvla\python.exe"
$PROJECT = "D:\Project\wrs-sealp"
$OUT = "$PROJECT\sealp\examples\layout\_output"
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$NAME = "tower_global_accel_$STAMP"
$LOGDIR = "$PROJECT\logs\global_accel_$STAMP"
$LOG = "$LOGDIR\global_accel.log"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
New-Item -ItemType Directory -Force $OUT | Out-Null

Write-Host "============================================================"
Write-Host "Global Accelerated Search"
Write-Host "Output name : $NAME"
Write-Host "Log         : $LOG"
Write-Host "============================================================"

$Args = @(
    "-m", "sealp.examples.layout.find_optimal_initial_layout_tower_global_accel",
    "--asmdef", "$PROJECT\sealp\assembly_sequence\_demo_output\topdown_tower.asmdef",
    "--config", "$PROJECT\sealp\config\sample_config.yaml",
    "--grasp-dir", "$PROJECT\sealp\examples\grasp\tower_grasp",
    "--output-dir", $OUT,
    "--output-name", $NAME,
    "--seed", "0",
    "--cdprim-type", "box",
    "--planner-obstacle-mode", "staging_aware",
    "--w-grasp", "0.25",
    "--w-manip", "0.50",
    "--w-dist", "0.05",
    "--w-rot", "0.20",
    "--global-explore", ([string]$Explore),
    "--global-elite", "3",
    "--global-refine-steps", "0.03,0.015,0.008",
    "--global-refine-rounds", "2",
    "--global-max-evals", ([string]$MaxEvals),
    "--global-accel-explore", ([string]$Explore),
    "--global-accel-legacy-replay", ([string]$LegacyReplay),
    "--global-accel-legacy-seed", "0",
    "--global-accel-search-grasp-cap", ([string]$SearchGraspCap),
    "--global-accel-authoritative-grasp-cap", "350",
    "--global-accel-rescore-topk", "8",
    "--global-accel-full-certify-topk", "0",
    "--global-accel-halton-attempts", "180",
    "--global-accel-coverage-bins", "4"
)

& $PY @Args 2>&1 |
    ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        Add-Content -Path $LOG -Value $line -Encoding UTF8
    }

$ExitCode = $LASTEXITCODE
Write-Host "============================================================"
Write-Host "Exit code: $ExitCode"
Write-Host "Log      : $LOG"
Write-Host "============================================================"
exit $ExitCode
