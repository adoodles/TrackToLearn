#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=
WORK_DATASET_FOLDER=

VALIDATION_SUBJECT_ID=ismrm2015
SUBJECT_ID=ismrm2015
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID}/scoring_data

mkdir -p $WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}

echo "Transfering data to working folder..."
cp -r ${DATASET_FOLDER}/datasets/${SUBJECT_ID} ${WORK_DATASET_FOLDER}/datasets/
cp -r ${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID} ${WORK_DATASET_FOLDER}/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
validation_dataset_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/${VALIDATION_SUBJECT_ID}.hdf5
reference_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/masks/${VALIDATION_SUBJECT_ID}_wm.nii.gz

max_ep=1000 # Chosen empirically
log_interval=50 # Log at n steps
lr=0.00005 # Learning rate
gamma=0.85 # Gamma for reward discounting

prob=0.0 # Noise to add to make a prob output. 0 for deterministic

npv=2 # Seed per voxel
theta=30 # Maximum angle for streamline curvature
n_actor=50000
length_weighting=0.1

EXPERIMENT=SAC_Auto_ISMRM2015TrainLength075

ID=$(date +"%F-%H_%M_%S")

seeds=(1111 2222 3333 4444 5555)

for rng_seed in "${seeds[@]}"
do

  DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

  python TrackToLearn/runners/sac_auto_train.py \
    "$DEST_FOLDER" \
    "$EXPERIMENT" \
    "$ID" \
    "${dataset_file}" \
    "${SUBJECT_ID}" \
    "${validation_dataset_file}" \
    "${VALIDATION_SUBJECT_ID}" \
    "${reference_file}" \
    "${SCORING_DATA}" \
    --max_ep=${max_ep} \
    --log_interval=${log_interval} \
    --lr=${lr} \
    --gamma=${gamma} \
    --rng_seed=${rng_seed} \
    --npv=${npv} \
    --theta=${theta} \
    --prob=$prob \
    --length_weighting=${length_weighting} \
    --use_gpu \
    --use_comet \
    --run_tractometer \
    --n_actor=${n_actor}
    # --render

    mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
    mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
    cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done
