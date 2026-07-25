# 5-fold cross-validation for one challenge on the local RTX 4070 (Windows, 12 GB).
# Mirrors run_cv.sh but with the 4070 config: batch 1, gradient checkpointing ON,
# num_workers 0 (Windows can't fork the augmentation closure). Use -Folds to run a
# subset, so the 4070 and the L4 can split the 15 fold-runs between them with no overlap.
#
#   powershell -ExecutionPolicy Bypass -File run_cv.ps1 -Challenge PED
#   powershell -ExecutionPolicy Bypass -File run_cv.ps1 -Challenge GLI -Folds 3,4
#
# Each fold resumes from its own checkpoint, so it is safe to stop/restart.

param(
  [Parameter(Mandatory=$true)][ValidateSet("GLI","MEN","PED")][string]$Challenge,
  [int[]]$Folds = @(0,1,2,3,4)
)

$ErrorActionPreference = "Continue"
$py = "C:\Users\sammi\anaconda3\envs\machine-learning-env\python.exe"
Set-Location "C:\Users\sammi\Desktop\projects\brats-2023"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS  = "ignore"

switch ($Challenge) {
  "GLI" { $DATA = "dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"; $LR = "1e-4"; $WD = "1e-5"; $FG = "0.9" }
  "MEN" { $DATA = "dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train"; $LR = "1e-4"; $WD = "1e-5"; $FG = "0.9" }
  "PED" { $DATA = "dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"; $LR = "5e-5"; $WD = "1e-4"; $FG = "0.95" }
}
$SPLIT = "splits/$($Challenge)_5fold_split.json"

foreach ($FOLD in $Folds) {
  $OUT = "output/$($Challenge)_swin_fold$($FOLD)"
  Write-Host "`n=== $Challenge fold $FOLD : training (4070, batch 1 + checkpointing) ===" -ForegroundColor Cyan
  & $py train_swin_unetr.py --model swin --challenge $Challenge --data_dir $DATA `
      --split_path $SPLIT --fold $FOLD --cache_dir "cache/$Challenge" --out_dir $OUT `
      --epochs 300 --lr $LR --weight_decay $WD --fg_prob $FG --backup_interval 100 `
      --batch_size 1 --num_workers 0

  if ($LASTEXITCODE -eq 0) {
    Write-Host "=== $Challenge fold $FOLD : full-volume evaluation ===" -ForegroundColor Cyan
    & $py evaluate_fullvol.py --challenge $Challenge --data_dir $DATA `
        --split_path $SPLIT --fold $FOLD --ckpt_path "$OUT/latest_ckpt.pth.tar" `
        --cache_dir "cache/$Challenge" --out_dir $OUT
  } else {
    Write-Host "fold $FOLD training exited with code $LASTEXITCODE - skipping its eval." -ForegroundColor Yellow
  }
}

Write-Host "`n=== aggregating $Challenge (folds present on this machine) ===" -ForegroundColor Cyan
& $py aggregate_cv.py --challenge $Challenge
