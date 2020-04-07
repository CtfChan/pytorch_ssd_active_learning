#!/bin/bash

# with weight reset
mkdir -p results/no_weight_reinit
for value in {1..3}
do
    for af in QBC RANDOM MEAN_STD ENTROPY MEAN_STD_WITH_BBOX LOCALIZATION_STABILITY VAR_RATIO MARGIN_SAMPLING BALD
    do
       python3 active_train.py --trial_number $value --acquisition_function $af --lr 1e-03 \
                --reset_weight False --save_dir ./results/no_weight_reinit/
    done
done
                               

# with weight reset
mkdir -p results/weight_reinit
for value in {1..3}
do
    for af in QBC RANDOM MEAN_STD ENTROPY MEAN_STD_WITH_BBOX LOCALIZATION_STABILITY VAR_RATIO MARGIN_SAMPLING BALD
    do
        python3 active_train.py --trial_number $value --acquisition_function $af --lr 1e-03 \
                --reset_weight True --save_dir ./results/weight_reinit/
    done
done
             

echo All done

