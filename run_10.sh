#!/usr/bin/env bash
set -euo pipefail

# ==== 사용자 설정 ====
PY=python
CMD="examples/spotless_trainer_maskadaptation.py"
DATA_DIR="examples/datasets/data/3DPW/outdoors_crosscountry_00/undistorted/"
MASK_DIR="examples/datasets/data/3DPW/outdoors_crosscountry_00/undistorted/binary_mask/"
BASE_OUT="output/test/maskadaptation9-cosblending_with_nodino"
DATA_FACTOR=4
TRAIN_KEY="clutter"
TEST_KEY="extra"

# ==== 로그 마커 (코드 출력과 맞춰 필요시 수정) ====
# Unmasked: 해당 마커 '다음 줄' 에 "PSNR: ..., SSIM: ..., LPIPS: ..." 라인이 온다고 가정
TRAIN_UNMASK_MARK="[EVAL] train rendered image save and train metrics"
TEST_UNMASK_MARK="[EVAL] saving TEST rendered image and test metrics"
TEST_MASK_MARK="[EVAL] TEST rendered image WITH MASK metrics"

# ==== 합계 변수 ====
# TRAIN (unmasked)
sum_tr_psnr=0;  sum_tr_ssim=0;  sum_tr_lpips=0;   n_tr_unmask=0
# TEST (unmasked)
sum_te_psnr=0;  sum_te_ssim=0;  sum_te_lpips=0;   n_te_unmask=0
# TEST (masked)
sum_te_mpsnr=0; sum_te_mssim=0; sum_te_mlpips=0;  n_te_mask=0

ok_runs=0

parse_unmasked_after_marker () {
  # 사용법: parse_unmasked_after_marker <LOGFILE> "<MARKER>"
  # 반환: stdout에 "psnr ssim lpips" (공백 구분) 출력. 실패시 빈 문자열
  local _log="$1"
  local _mark="$2"
  local _lineno
  _lineno=$(grep -nF "${_mark}" "${_log}" | tail -n1 | cut -d: -f1 || true)
  if [[ -z "${_lineno}" ]]; then
    echo ""
    return 0
  fi
  local _next=$((_lineno+1))
  local _line
  _line=$(sed -n "${_next}p" "${_log}" || true)
  if [[ -z "${_line}" ]]; then
    echo ""
    return 0
  fi
  # 예: "PSNR: 27.123456789, SSIM: 0.902345678, LPIPS: 0.134567890"
  local _psnr _ssim _lpips
  _psnr=$(echo "${_line}"  | sed -E 's/.*PSNR: ([0-9.]+).*/\1/') || true
  _ssim=$(echo "${_line}"  | sed -E 's/.*SSIM: ([0-9.]+).*/\1/') || true
  _lpips=$(echo "${_line}" | sed -E 's/.*LPIPS: ([0-9.]+).*/\1/') || true
  if [[ -z "${_psnr}" || -z "${_ssim}" || -z "${_lpips}" ]]; then
    echo ""
  else
    echo "${_psnr} ${_ssim} ${_lpips}"
  fi
}

parse_masked_inline () {
  # 사용법: parse_masked_inline <LOGFILE> "<MARKER>"
  # 반환: stdout에 "mpsnr mssim mlpips" (공백 구분) 출력. 실패시 빈 문자열
  local _log="$1"
  local _mark="$2"
  local _line
  _line=$(grep -F "${_mark}" "${_log}" | tail -n1 || true)
  if [[ -z "${_line}" ]]; then
    echo ""
    return 0
  fi
  # 예: "MASK_PSNR: 28.123456789, MASK_SSIM: 0.912345678, MASK_LPIPS: 0.123456789"
  local _mpsnr _mssim _mlpips
  _mpsnr=$(echo "${_line}" | sed -E 's/.*MASK_PSNR: ([0-9.]+).*/\1/') || true
  _mssim=$(echo "${_line}" | sed -E 's/.*MASK_SSIM: ([0-9.]+).*/\1/') || true
  _mlpips=$(echo "${_line}"| sed -E 's/.*MASK_LPIPS: ([0-9.]+).*/\1/') || true
  if [[ -z "${_mpsnr}" || -z "${_mssim}" || -z "${_mlpips}" ]]; then
    echo ""
  else
    echo "${_mpsnr} ${_mssim} ${_mlpips}"
  fi
}

for i in $(seq 1 10); do
  OUTDIR="${BASE_OUT}_run${i}"
  LOG="${OUTDIR}/train_eval.log"
  mkdir -p "${OUTDIR}"

  echo ">>> [RUN ${i}] result_dir=${OUTDIR}"

  # 실행(실시간 콘솔+파일)
  if ! stdbuf -oL -eL ${PY} -u "${CMD}" \
      --data_dir "${DATA_DIR}" \
      --data_factor "${DATA_FACTOR}" \
      --loss_type robust \
      --semantics \
      --no-cluster \
      --train_keyword "${TRAIN_KEY}" \
      --test_keyword "${TEST_KEY}" \
      --mask_dir "${MASK_DIR}" \
      --result_dir "${OUTDIR}" \
      2>&1 | tee "${LOG}"; then
    echo "[WARN] Run ${i} failed. See ${LOG}"
    continue
  fi

  ok_runs=$((ok_runs+1))

  # ===== TRAIN (unmasked) =====
  tr_unmasked="$(parse_unmasked_after_marker "${LOG}" "${TRAIN_UNMASK_MARK}")"
  if [[ -n "${tr_unmasked}" ]]; then
    read -r tr_psnr tr_ssim tr_lpips <<< "${tr_unmasked}"
    sum_tr_psnr=$(echo "${sum_tr_psnr} + ${tr_psnr}" | bc -l)
    sum_tr_ssim=$(echo "${sum_tr_ssim} + ${tr_ssim}" | bc -l)
    sum_tr_lpips=$(echo "${sum_tr_lpips} + ${tr_lpips}" | bc -l)
    n_tr_unmask=$((n_tr_unmask+1))
  else
    echo "[WARN] Run ${i}: TRAIN (unmasked) metrics not found in ${LOG}"
  fi

  # ===== TEST (unmasked) =====
  te_unmasked="$(parse_unmasked_after_marker "${LOG}" "${TEST_UNMASK_MARK}")"
  if [[ -n "${te_unmasked}" ]]; then
    read -r te_psnr te_ssim te_lpips <<< "${te_unmasked}"
    sum_te_psnr=$(echo "${sum_te_psnr} + ${te_psnr}" | bc -l)
    sum_te_ssim=$(echo "${sum_te_ssim} + ${te_ssim}" | bc -l)
    sum_te_lpips=$(echo "${sum_te_lpips} + ${te_lpips}" | bc -l)
    n_te_unmask=$((n_te_unmask+1))
  else
    echo "[WARN] Run ${i}: TEST (unmasked) metrics not found in ${LOG}"
    # 계속 진행
  fi

  # ===== TEST (masked) =====
  te_masked="$(parse_masked_inline "${LOG}" "${TEST_MASK_MARK}")"
  if [[ -n "${te_masked}" ]]; then
    read -r te_mpsnr te_mssim te_mlpips <<< "${te_masked}"
    sum_te_mpsnr=$(echo "${sum_te_mpsnr} + ${te_mpsnr}" | bc -l)
    sum_te_mssim=$(echo "${sum_te_mssim} + ${te_mssim}" | bc -l)
    sum_te_mlpips=$(echo "${sum_te_mlpips} + ${te_mlpips}" | bc -l)
    n_te_mask=$((n_te_mask+1))
  else
    echo "[WARN] Run ${i}: TEST (masked) metrics not found in ${LOG}"
  fi

  # 개별 결과 저장
  {
    echo "TRAIN_PSNR ${tr_psnr:-NA}"
    echo "TRAIN_SSIM ${tr_ssim:-NA}"
    echo "TRAIN_LPIPS ${tr_lpips:-NA}"
    echo "TEST_PSNR ${te_psnr:-NA}"
    echo "TEST_SSIM ${te_ssim:-NA}"
    echo "TEST_LPIPS ${te_lpips:-NA}"
    echo "TEST_MASK_PSNR ${te_mpsnr:-NA}"
    echo "TEST_MASK_SSIM ${te_mssim:-NA}"
    echo "TEST_MASK_LPIPS ${te_mlpips:-NA}"
  } > "${OUTDIR}/metrics_last_eval.txt"

done

if [[ "${ok_runs}" -eq 0 ]]; then
  echo "[ERROR] No successful runs. Abort."
  exit 1
fi

# ===== 평균 계산 (존재한 항목에 대해서만) =====
avg_tr_psnr=$( [[ $n_tr_unmask -gt 0 ]] && echo "${sum_tr_psnr} / ${n_tr_unmask}" | bc -l || echo "nan" )
avg_tr_ssim=$( [[ $n_tr_unmask -gt 0 ]] && echo "${sum_tr_ssim} / ${n_tr_unmask}" | bc -l || echo "nan" )
avg_tr_lpips=$( [[ $n_tr_unmask -gt 0 ]] && echo "${sum_tr_lpips} / ${n_tr_unmask}" | bc -l || echo "nan" )

avg_te_psnr=$( [[ $n_te_unmask -gt 0 ]] && echo "${sum_te_psnr} / ${n_te_unmask}" | bc -l || echo "nan" )
avg_te_ssim=$( [[ $n_te_unmask -gt 0 ]] && echo "${sum_te_ssim} / ${n_te_unmask}" | bc -l || echo "nan" )
avg_te_lpips=$( [[ $n_te_unmask -gt 0 ]] && echo "${sum_te_lpips} / ${n_te_unmask}" | bc -l || echo "nan" )

avg_te_mpsnr=$( [[ $n_te_mask -gt 0 ]] && echo "${sum_te_mpsnr} / ${n_te_mask}" | bc -l || echo "nan" )
avg_te_mssim=$( [[ $n_te_mask -gt 0 ]] && echo "${sum_te_mssim} / ${n_te_mask}" | bc -l || echo "nan" )
avg_te_mlpips=$( [[ $n_te_mask -gt 0 ]] && echo "${sum_te_mlpips} / ${n_te_mask}" | bc -l || echo "nan" )

printf "\n===== Averages over %d successful runs =====\n" "${ok_runs}"
printf "[TRAIN] averaged(10) metrics" 
printf "[TRAIN]        PSNR       : %s dB\n" "${avg_tr_psnr}"
printf "[TRAIN]        SSIM       : %s\n"     "${avg_tr_ssim}"
printf "[TRAIN]        LPIPS      : %s\n"      "${avg_tr_lpips}"

printf "[TEST] averaged(10) metrics" 
printf "[TEST]        PSNR       : %s dB\n" "${avg_te_psnr}"
printf "[TEST]        SSIM       : %s\n"     "${avg_te_ssim}"
printf "[TEST]        LPIPS      : %s\n"      "${avg_te_lpips}"

printf "[MASKED-TEST] averaged(10) metrics" 
printf "[MASKED-TEST]        PSNR       : %s dB\n" "${avg_te_psnr}"
printf "[MASKED-TEST]        SSIM       : %s\n"     "${avg_te_ssim}"
printf "[MASKED-TEST]        LPIPS      : %s\n"      "${avg_te_lpips}"
