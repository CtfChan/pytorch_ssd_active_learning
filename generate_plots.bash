#!/bin/bash


# Cmd do queue jobs on one GPU
# # add QBC last - it's so slow
# for af in RANDOM LOCALIZATION_STABILITY MEAN_STD ENTROPY BALD VAR_RATIO MARGIN_SAMPLING
# #for af in QBC
# do
    # echo $af
    #for value in {1..3}
    # do
    #     echo $value
    #     python3 active_train.py --trial_number $value --acquisition_function $af
    # done
    # python3 active_train.py --trial_number 1 --acquisition_function $af

# done
# echo All done


# # two GPUs
# for af in RANDOM LOCALIZATION_STABILITY MEAN_STD ENTROPY BALD VAR_RATIO MARGIN_SAMPLING
# do
#     python3 active_train.py --trial_number 1 --acquisition_function $af --lr 1e-04 --save_dir ./results/1e-04/
# done

# for af in RANDOM LOCALIZATION_STABILITY MEAN_STD ENTROPY BALD VAR_RATIO MARGIN_SAMPLING
# do
#     python3 active_train.py --trial_number 1 --acquisition_function $af --lr 1e-05 --save_dir ./results/1e-05/
# done


#for af in RANDOM MEAN_STD ENTROPY BALD VAR_RATIO MARGIN_SAMPLING MEAN_STD_WITH_BBOX LOCALIZATION_STABILITY
#for af in MARGIN_SAMPLING
for af in QBC RANDOM LOCALIZATION_STABILITY MEAN_STD_WITH_BBOX
do
    python3 active_train.py --trial_number 1 --acquisition_function $af --lr 1e-03 --save_dir ./results/
done



echo All done

