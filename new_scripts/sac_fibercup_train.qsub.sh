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

# This should point to your dataset folder
DATASET_FOLDER='/home/awuxingh/data/fibercup'

# BELOW SOME PARAMETERS THAT DEPEND ON MY FILE STRUCTURE
# YOU CAN CHANGE ANYTHING AS YOU WISH

# Should be relatively stable
VALIDATION_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=./experiments
SCORING_DATA=${DATASET_FOLDER}/scoring_data

# Data params
dataset_file=$DATASET_FOLDER/${SUBJECT_ID}.hdf5
validation_dataset_file=$DATASET_FOLDER/${VALIDATION_SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/masks/${SUBJECT_ID}_wm.nii.gz

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

mkdir -p ./experiments
mkdir -p ./experiments/$EXPERIMENT

ID=$(date +"%F-%H_%M_%S")

DEST_FOLDER=$EXPERIMENTS_FOLDER/$EXPERIMENT/$ID
BASE_FOLDER='/home/awuxingh/new_TTL/TrackToLearn'

if (( $CUDA_VISIBLE_DEVICES > -1 )); then

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

n_seeds_per_voxel=33
min_length=20
max_length=200

valid_noise=0.1
python3 $BASE_FOLDER/TrackToLearn/runners/ttl_validation.py $DEST_FOLDER \
  "$EXPERIMENT" \
  "$ID" \
  "${validation_dataset_file}" \
  "${VALIDATION_SUBJECT_ID}" \
  "${reference_file}" \
  $DEST_FOLDER/model \
  $DEST_FOLDER/model/hyperparameters.json \
  --prob="${valid_noise}" \
  --npv="${n_seeds_per_voxel}" \
  --min_length="$min_length" \
  --max_length="$max_length" \
  --use_gpu \
  --fa_map="$DATASET_FOLDER"/dti/"${SUBJECT_ID}"_fa.nii.gz \
  --remove_invalid_streamlines

mkdir -p $DEST_FOLDER/scoring_"${valid_noise}"_fa

python3 $BASE_FOLDER/scripts/score_tractogram.py $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${VALIDATION_SUBJECT_ID}".trk \
  $SCORING_DATA \
  $DEST_FOLDER/scoring_"${valid_noise}"_fa \
  --compute_ic_ib \
  --save_full_vc \
  --save_full_ic \
  --save_full_nc \
  --save_ib \
  --save_vb -f -v
fi
