#$ -S /bin/bash
#$ -l tmem=20G

#$ -R y
#$ -j y
#$ -l gpu=true
#$ -N TTL_train
#$ -l gpu_type=gtx1080ti
#$ -l h_rt=1:0:0
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

max_ep=1000 # Chosen empirically
log_interval=500 # Log at n steps
lr=8.56e-6 # Learning rate 
gamma=0.776 # Gamma for reward discounting
rng_seed=4033 # Seed for general randomness

# TD3 parameters
action_std=0.334 # STD deviation for action

# Env parameters
n_seeds_per_voxel=2 # Seed per voxel
max_angle=60 # Maximum angle for streamline curvature

EXPERIMENT=fibercup

mkdir -p ./experiments
mkdir -p ./experiments/$EXPERIMENT

ID="to_score"

DEST_FOLDER=$EXPERIMENTS_FOLDER/$EXPERIMENT/$ID
BASE_FOLDER='/home/awuxingh/new_TTL/TrackToLearn'

if (( $CUDA_VISIBLE_DEVICES > -1 )); then
valid_noise=0.1
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
