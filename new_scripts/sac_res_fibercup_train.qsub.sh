#$ -S /bin/bash
#$ -l tmem=20G

#$ -R y
#$ -j y
#$ -l gpu=true
#$ -N TTL_train
#$ -l gpu_type=gtx1080ti
#$ -l h_rt=24:00:0
#$ -cwd

source /share/apps/source_files/cuda/cuda-11.2.source
source /share/apps/source_files/python/python-3.8.5.source
nvidia-smi

# BELOW SOME PARAMETERS THAT DEPEND ON MY FILE STRUCTURE
# YOU CAN CHANGE ANYTHING AS YOU WISH

# Should be relatively stable
VALIDATION_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=./experiments
SCORING_DATA=${DATASET_FOLDER}/scoring_data


# RL params

max_ep=80 # Chosen empirically
log_interval=10 # Log at n steps
lr=4.35e-4 # Learning rate 
gamma=0.9 # Gamma for reward discounting
rng_seed=4033 # Seed for general randomness

# SAC parameters
alpha=0.087

# Env parameters
n_seeds_per_voxel=10 # Seed per voxel
max_angle=30 # Maximum angle for streamline curvature

EXPERIMENT=fibercup
RESOLUTION=('fibercup_3mm' 'fibercup_3.5mm' 'fibercup_4mm')

mkdir -p ./experiments
mkdir -p ./experiments/$EXPERIMENT

ID=$(date +"%F-%H_%M_%S")

DEST_FOLDER=$EXPERIMENTS_FOLDER/$EXPERIMENT/$ID
BASE_FOLDER='/home/awuxingh/new_TTL/TrackToLearn'

if (( $CUDA_VISIBLE_DEVICES > -1 )); then

for res_train in "${RESOLUTION[@]}"
do 
  # This should point to your dataset folder
  DATASET_FOLDER="/home/awuxingh/data/${res_train}"
  VALIDATION_SUBJECT_ID=$res_train
  SUBJECT_ID=$res_train

  # Data params
  dataset_file=$DATASET_FOLDER/${res_train}.hdf5
  validation_dataset_file=$DATASET_FOLDER/${res_train}.hdf5
  reference_file=$DATASET_FOLDER/masks/fibercup_wm.nii.gz

  python3 $BASE_FOLDER/TrackToLearn/trainers/sac_train.py \
    $DEST_FOLDER \
    $EXPERIMENT \
    $ID \
    ${dataset_file} \
    ${SUBJECT_ID} \
    ${validation_dataset_file} \
    ${VALIDATION_SUBJECT_ID} \
    ${reference_file} \
    ${SCORING_DATA} \
    --max_ep=${max_ep} \
    --log_interval=${log_interval} \
    --alpha=${alpha} \
    --lr=${lr} \
    --gamma=${gamma} \
    --rng_seed=${rng_seed} \
    --theta=${max_angle} \
    --npv=${n_seeds_per_voxel} \
    --use_gpu \
    --run_tractometer \
    --use_comet

done
fi
