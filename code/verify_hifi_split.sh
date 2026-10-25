#!/usr/bin/env bash
# verify_hifi_split.sh
# 校验公开仓 JielunPeng/HiFi-AVDF (6 zip + 6 csv) 解压后是否与本机 D:\Datasets\HiFi-AVDF 划分一致。
# 评测以 6 个 per-gen csv 为权威清单 (本机评测即 join 各 gen csv 的 overall_label)。
# 用法:
#   BASE=/root/autodl-tmp/SyncX/Datasets/HiFi-AVDF bash verify_hifi_split.sh
# 关键正确性点:
#   * 数据行数用 awk 'END{print NR-1}' 统计 —— 文件末行无换行符时 wc -l 会少算 1 行, 必须用 awk。
#   * 解压防嵌套: 若 zip 内顶层目录已是 gen 名, 解压到根目录即可; 否则拉平, 保证视频落在 $g/ 下,
#     与 csv 的 "kling2.5/1.mp4" 相对路径直接命中。
set -uo pipefail

BASE="${BASE:-/root/autodl-tmp/SyncX/Datasets/HiFi-AVDF}"
cd "$BASE" || { echo "无法进入 $BASE"; exit 1; }

# ---- 本机基准 (准确数据行, awk 统计; 末行无换行也正确) ----
declare -A EXPECT=(
  [kling2.5]=460 [pixverseV5]=616 [seedance1.0]=788
  [sora2]=728 [veo3.1]=638 [wan2.5]=580
)
HEADER="video_path,audio_label,visual_label,overall_label"
GENS="kling2.5 pixverseV5 seedance1.0 sora2 veo3.1 wan2.5"

echo "===== Step B: 解压 (防嵌套, 已解压则跳过) ====="
for g in $GENS; do
  if [ -d "$g" ] && [ "$(find "$g" -maxdepth 1 -type f -iname '*.mp4' 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[skip] $g 已解压, 跳过"
    continue
  fi
  if [ ! -f "$g.zip" ]; then echo "[WARN] $g.zip 不存在, 跳过解压"; continue; fi
  rm -rf "extract_tmp_$g"; mkdir -p "extract_tmp_$g"
  if command -v unzip >/dev/null 2>&1; then
    unzip -oq "$g.zip" -d "extract_tmp_$g"
  else
    python3 - <<PY
import zipfile
zipfile.ZipFile("$g.zip").extractall("extract_tmp_$g")
PY
  fi
  # 防嵌套: 若 extract_tmp_$g/$g 存在则下钻一层, 否则用 extract_tmp_$g 本身
  src="extract_tmp_$g"
  [ -d "$src/$g" ] && src="$src/$g"
  mkdir -p "$g"
  mv "$src"/* "$g"/ 2>/dev/null || true
  rm -rf "extract_tmp_$g"
  echo "[done] $g 解压完成 -> $g/"
done

echo ""
echo "===== Step C: 核对划分 (与本机等价) ====="
ALL_OK=1

# C1: 表头一致 + 准确数据行 == 基准
for g in $GENS; do
  [ ! -f "$g.csv" ] && { echo "[FAIL] $g.csv 缺失"; ALL_OK=0; continue; }
  hdr=$(head -n1 "$g.csv")
  rows=$(awk 'END{print NR-1}' "$g.csv")
  hdr_ok=0; [ "$hdr" = "$HEADER" ] && hdr_ok=1
  rows_ok=0; [ "$rows" -eq "${EXPECT[$g]}" ] && rows_ok=1
  [ "$hdr_ok" -eq 0 ] && { echo "[FAIL] $g.csv 表头不一致: [$hdr]"; ALL_OK=0; }
  [ "$rows_ok" -eq 0 ] && { echo "[FAIL] $g.csv 数据行=$rows 期望 ${EXPECT[$g]}"; ALL_OK=0; }
  echo "[C1] $g 表头=$([ $hdr_ok -eq 1 ] && echo OK || echo BAD) 数据行=$rows/期望${EXPECT[$g]}"
done

# C2: 解压视频数 == 准确数据行
for g in $GENS; do
  rows=$(awk 'END{print NR-1}' "$g.csv")
  n_vid=$(find "$g" -maxdepth 1 -type f 2>/dev/null | grep -Ei '[.](mp4|avi|mov|mkv)$' | wc -l)
  if [ "$n_vid" -ne "$rows" ]; then echo "[FAIL] $g 视频数=$n_vid != 数据行=$rows"; ALL_OK=0; fi
  echo "[C2] $g 视频数=$n_vid / 数据行=$rows"
done

# C3: csv 中每个 video_path 文件必须存在 (评测能吃到的样本)
#     用 awk+comm, 避开 while-read 对"末行无换行"丢行的陷阱
for g in $GENS; do
  awk -F, 'NR>1 && $1!="" && $1!="video_path"{print $1}' "$g.csv" | sort -u > /tmp/csv_${g}.txt
  find "$g" -maxdepth 1 -type f 2>/dev/null | grep -Ei '[.](mp4|avi|mov|mkv)$' | sort -u > /tmp/dir_${g}.txt
  total=$(wc -l < /tmp/csv_${g}.txt)
  missing=$(comm -23 /tmp/csv_${g}.txt /tmp/dir_${g}.txt | wc -l)   # 在csv不在文件夹(致命)
  extra=$(comm -13 /tmp/csv_${g}.txt /tmp/dir_${g}.txt | wc -l)     # 在文件夹不在csv(信息性)
  if [ "$missing" -ne 0 ]; then echo "[FAIL] $g 缺失 $missing/$total 个文件"; ALL_OK=0; fi
  echo "[C3] $g 文件存在性: $((total-missing))/$total 存在 (文件夹多出 $extra 个非csv文件, 信息性)"
done

# 重建等价 master (从 6 gen csv 生成 union, 规模与官方 master.jsonl 一致 3810)
total_all=0
for g in $GENS; do total_all=$((total_all + $(awk 'END{print NR-1}' "$g.csv"))); done
echo ""
echo "[rebuild] 6 gen 数据行合计 = $total_all (官方 face_verify_hifi_master.jsonl = 3810)"

echo ""
if [ "$ALL_OK" -eq 1 ]; then
  echo "=== PASS: 公开仓与本机 per-gen 划分完全一致, 评测可直接吃 per-gen csv 的 overall_label ==="
else
  echo "=== FAIL: 请检查上方 [FAIL] ==="
  exit 1
fi
