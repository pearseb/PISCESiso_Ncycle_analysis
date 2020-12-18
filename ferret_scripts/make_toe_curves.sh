#!/bin/bash

export WERK=/mnt/lustre/users/pearseb/NEMO_OUT/analysis_d15N_data
cd $WERK

rm ToE_picndep_curves.txt ToE_fut_curves.txt ToE_futndep_curves.txt

for year in $(seq 1850 1 2100); do
  echo ${year}
  ferret -script /users/pearseb/analysis_d15N_scripts/calculate_toe_percentcover.jnl ${year}
done
