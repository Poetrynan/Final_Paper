<#
clear_prep_caches.ps1 — 安全清除 Sync-X 已抽取的特征缓存（prep/）

用途：
  v_lip 维度已锁定为 192-D（64 口周点）。所有旧契约抽取的特征缓存（.pt）必须清除后，
  用 prep_*.py 重新抽取。本脚本只删除各数据集下的 <ROOT>/prep 子目录（特征缓存），
  绝不触碰原始视频/音频数据。

安全设计：
  1) 默认【只读报告】模式：只打印每个 prep 目录是否存在、.pt 数量、占用大小，不删除任何东西。
  2) 仅当显式带 -Commit 参数时才执行删除。
  3) 删除走【回收站】(Recycle Bin)，非永久删除 —— 误删可在回收站恢复。
     （这些缓存本就可由 prep_*.py 重新生成，故无需另行备份。）
  4) 只针对下方 $PrepRoots 列出的 prep 子目录；对原始数据集目录零影响。

用法（在 PowerShell 中，建议先在项目 code/ 目录）：
  .\clear_prep_caches.ps1            # ① 先看报告，确认范围无误
  .\clear_prep_caches.ps1 -Commit    # ② 确认无误后再真正清除（进回收站）

注意：
  - 回收站空间不足时，SendToRecycleBin 会失败；如确需永久删除，把下方
    DeleteDirectory 的 'SendToRecycleBin' 改为 'DeletePermanently'（谨慎！）。
  - 本脚本只删 <ROOT>/prep；若某数据集尚未抽取（无 prep 目录），会自动跳过。
#>

[CmdletBinding()]
param(
    [switch]$Commit = $false
)

# ── 10 个数据集的特征缓存根（与 prep_*.py 的 PREP 定义一一对应）──────────────
$PrepRoots = @(
    "D:/Datasets/DeepSpeak_v1_1/clips/prep",
    "D:/Datasets/Celeb-DF-v2/prep",
    "D:/Datasets/Deepfake-Eval-2024/prep",
    "D:/Datasets/FakeAVCeleb_v1.2_split/prep",
    "D:/Datasets/HiFi-AVDF/prep",
    "D:/Datasets/MVAD/prep",
    "D:/Datasets/TalkingHeadBench/prep",
    "D:/Datasets/VoxCeleb2_subset/prep",
    "D:/DiffusionBenchmark/prep",
    "D:/Datasets/WildDeepfake/prep"
)

# 回收站删除（可逆）
Add-Type -AssemblyName Microsoft.VisualBasic
function Send-ToRecycleBin($path) {
    [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(
        $path, 'OnlyErrorDialogs', 'SendToRecycleBin')
}

Write-Host ""
if (-not $Commit) {
    Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  【只读报告模式】未带 -Commit，不会删除任何文件。" -ForegroundColor Yellow
    Write-Host "   确认下方范围无误后，再运行: .\clear_prep_caches.ps1 -Commit" -ForegroundColor Yellow
    Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
} else {
    Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Red
    Write-Host "  【执行模式 -Commit】将把以下存在的 prep 目录送入回收站（可逆）。" -ForegroundColor Red
    Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Red
}

$totalPt    = 0
$totalBytes = [long]0
$removed    = 0

foreach ($root in $PrepRoots) {
    if (-not (Test-Path $root)) {
        Write-Host ("  [跳过] 不存在: {0}" -f $root) -ForegroundColor DarkGray
        continue
    }
    $ptFiles = Get-ChildItem -Path $root -Recurse -Filter *.pt -File -ErrorAction SilentlyContinue
    $ptCount = $ptFiles.Count
    $size    = ($ptFiles | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $size) { $size = 0 }
    $totalPt    += $ptCount
    $totalBytes += $size
    $sizeMB = [math]::Round($size / 1MB, 1)

    Write-Host ("  [存在] {0}" -f $root) -ForegroundColor Cyan
    Write-Host ("         .pt 数量 = {0}   占用 ≈ {1} MB" -f $ptCount, $sizeMB)

    if ($Commit -and $ptCount -gt 0) {
        try {
            Send-ToRecycleBin $root
            Write-Host ("         ✅ 已送入回收站" -f $root) -ForegroundColor Green
            $removed++
        } catch {
            Write-Host ("         ❌ 删除失败: {0}" -f $_.Exception.Message) -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host ("汇总: 发现 .pt 共 {0} 个, 约 {1} MB" -f $totalPt, [math]::Round($totalBytes/1MB,1)) -ForegroundColor White
if ($Commit) {
    Write-Host ("已送回收站目录数: {0} / {1}" -f $removed, $PrepRoots.Count) -ForegroundColor White
    Write-Host "若需恢复：打开回收站还原即可。重新抽取请用各 prep_*.py。" -ForegroundColor DarkGray
} else {
    Write-Host "（以上为报告，未删除。确认无误后加 -Commit 执行。）" -ForegroundColor DarkGray
}
Write-Host ""
