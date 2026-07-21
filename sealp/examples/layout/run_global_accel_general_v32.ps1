param(
    [string]$Asmdef = "D:\Project\wrs-sealp\sealp\assembly_sequence\_demo_output\topdown_tower.asmdef",
    [string]$Config = "D:\Project\wrs-sealp\sealp\config\sample_config.yaml",
    [string]$GraspDir = "D:\Project\wrs-sealp\sealp\examples\grasp\tower_grasp",
    [string]$OutputPrefix = "global_accel_general_v32",
    [string]$WarmLayout = "",
    [int]$Explore = 80,
    [int]$MaxEvals = 220,
    [int]$SearchGraspCap = 160,
    [int]$AuthoritativeGraspCap = 350,
    [ValidateSet("same_grasp_progressive", "same_grasp_full")]
    [string]$GraspMode = "same_grasp_progressive",
    [ValidateSet("medoid", "extreme", "first")]
    [string]$FpsSeedMode = "medoid",
    [ValidateSet("fps", "legacy_uniform")]
    [string]$SearchSubsetMode = "fps",
    [ValidateSet("fps", "legacy_uniform")]
    [string]$AuthoritativeSubsetMode = "legacy_uniform",
    [switch]$EnableRefine
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$PY = "D:\Soft\tools\anaconda\envs\spatialvla\python.exe"
$PROJECT = "D:\Project\wrs-sealp"
$OUT = "$PROJECT\sealp\examples\layout\_output"
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$NAME = "${OutputPrefix}_$STAMP"
$CHECKPOINT = "$OUT\${OutputPrefix}_best_so_far.json"
$LOGDIR = "$PROJECT\logs\${OutputPrefix}_$STAMP"
$LOG = "$LOGDIR\global_accel_general_v32.log"

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
New-Item -ItemType Directory -Force $OUT | Out-Null

$Args = @(
    "-m", "sealp.examples.layout.find_optimal_initial_layout_tower_global_accel_general_v32",
    "--asmdef", $Asmdef,
    "--config", $Config,
    "--grasp-dir", $GraspDir,
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
    "--global-accel-legacy-replay", "0",
    "--global-accel-no-warm-first-refine",
    "--global-accel-search-grasp-cap", ([string]$SearchGraspCap),
    "--global-accel-authoritative-grasp-cap", ([string]$AuthoritativeGraspCap),
    "--global-accel-grasp-mode", $GraspMode,
    "--global-accel-fps-seed-mode", $FpsSeedMode,
    "--global-accel-search-subset-mode", $SearchSubsetMode,
    "--global-accel-authoritative-subset-mode", $AuthoritativeSubsetMode,
    "--global-accel-rescore-topk", "8",
    "--global-accel-full-certify-topk", "0",
    "--global-accel-halton-layout-attempts", "320",
    "--global-accel-halton-repair-attempts", "24",
    "--global-accel-coverage-bins", "4",
    "--global-accel-checkpoint", $CHECKPOINT
)

if (-not $EnableRefine) {
    $Args += "--no-refine"
}

if ($WarmLayout) {
    if (-not (Test-Path -LiteralPath $WarmLayout)) {
        throw "Warm layout file does not exist: $WarmLayout"
    }
    $ResolvedWarmLayout = (Resolve-Path -LiteralPath $WarmLayout).Path
    $Args += @("--global-accel-warm-layout", $ResolvedWarmLayout)
}

Write-Host "============================================================"
Write-Host "General Global Accelerated Search V3.2"
Write-Host "Search subset        : $SearchSubsetMode, cap=$SearchGraspCap"
Write-Host "Authoritative subset : $AuthoritativeSubsetMode, cap=$AuthoritativeGraspCap"
Write-Host "Refine               : $EnableRefine"
Write-Host "Max evals            : $MaxEvals"
Write-Host "Checkpoint           : $CHECKPOINT"
Write-Host "Log                  : $LOG"
Write-Host "============================================================"

& $PY @Args 2>&1 |
    ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        Add-Content -Path $LOG -Value $line -Encoding UTF8
    }

$ExitCode = $LASTEXITCODE
Write-Host "============================================================"
Write-Host "Exit code : $ExitCode"
Write-Host "Checkpoint: $CHECKPOINT"
Write-Host "Log       : $LOG"
Write-Host "============================================================"
exit $ExitCode
