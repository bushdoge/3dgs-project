#!/bin/bash
# C3（監査R-03）：thin − edge_other（選択性。eval_contrast_regions.py）を 床なしGT(4096spp) / 従来GT(64spp+デノイザ) の各runで測って1行ずつ出す。
# 使い方: bash research/wirebench/c3_selectivity.sh > tmp/c3_selectivity.txt（研究メモ §9ar-5）
cd /workspace
run(){ v=$(python3 research/wirebench/eval_contrast_regions.py experiments/$1 $3 $2 2>&1 | grep 'thin − edge_other' | grep -o '[-+][0-9.]*' | head -1); echo "$1 $2 $3 $v"; }
for e in 20260924_wb_r15mm_f80_open_clean 20260924_wb_r15mm_f80_open_res512_clean; do
  for r in 1 2 3; do run $e output_3dgs7k_r$r 7000; run $e output_3dgs30k_r$r 30000; run $e output_mip30k_r$r 30000; done; done
A=20260704_wb_r15mm_f80_open
for m in output_gen_newdef_r1 output_gen_newdef_r2 output_gen_newdef_r3; do run $A $m 7000; done
for m in output_3dgs30k_r1 output_3dgs30k_r2 output_3dgs30k_r3; do run $A $m 30000; done
for m in output_mip_30k output_mip_30k_rerun output_ksdec_01_r3; do run $A $m 30000; done
B=20260907_wb_r15mm_f80_open_res512
for m in output_3dgs7000 output_3dgs7000_r3 output_3dgs7000_r4; do run $B $m 7000; done
for m in output_3dgs30000 output_3dgs30000_r3 output_3dgs30000_r4; do run $B $m 30000; done
for m in output_mip30k_ks01 output_mip30k_ks01_r2 output_mip30k_ks01_r3; do run $B $m 30000; done
