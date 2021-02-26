#!/bin/bash
#SBATCH --gres=gpu:v100:1       # Request specific V100
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=150G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=4-00:00     # DD-HH:MM:SS
#SBATCH --mail-user=gareth.zhang@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python/3.7.4 cuda cudnn

SOURCEDIR=~/projects/def-tbarfoot/hzhang34/KPConvAnnot

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt

# Prepare data
SEQ_DIR=$SLURM_TMPDIR/boreas/raw/sequences
mkdir -p $SEQ_DIR

####################
# Training sequences
####################

DAY=boreas-2020-12-01-13-26
tar xvf ~/scratch/boreas/$DAY/$DAY.tar -C $SEQ_DIR
mkdir -p $SEQ_DIR/$DAY/velodyne
mv $SEQ_DIR/$DAY/*.ply $SEQ_DIR/$DAY/velodyne
cp ~/scratch/boreas/$DAY/map_poses.txt $SEQ_DIR/$DAY

DAY=boreas-2020-12-04-14-00
tar xvf ~/scratch/boreas/$DAY/$DAY.tar -C $SEQ_DIR
mkdir $SEQ_DIR/$DAY/velodyne
mv $SEQ_DIR/$DAY/*.ply $SEQ_DIR/$DAY/velodyne
cp ~/scratch/boreas/$DAY/map_poses.txt $SEQ_DIR/$DAY

DAY=boreas-2020-12-08-14-01
tar xvf ~/scratch/boreas/$DAY/$DAY.tar -C $SEQ_DIR
mkdir $SEQ_DIR/$DAY/velodyne
mv $SEQ_DIR/$DAY/*.ply $SEQ_DIR/$DAY/velodyne
cp ~/scratch/boreas/$DAY/map_poses.txt $SEQ_DIR/$DAY

DAY=boreas-2021-02-02-night
tar xvf ~/scratch/boreas/$DAY/$DAY.tar -C $SEQ_DIR
mkdir $SEQ_DIR/$DAY/velodyne
mv $SEQ_DIR/$DAY/*.ply $SEQ_DIR/$DAY/velodyne
cp ~/scratch/boreas/$DAY/map_poses.txt $SEQ_DIR/$DAY

###########################
# Test sequences for Boreas
###########################
DAY=boreas-2021-01-31-dufferin-00
tar xvf ~/scratch/boreas/$DAY/$DAY.tar -C $SEQ_DIR

#DAY=boreas-2021-01-31-highway-00-00
#tar xvf ~/scratch/boreas/$DAY/$DAY.tar -C $SEQ_DIR
#
#DAY=boreas-2021-01-31-local-00
#tar xvf ~/scratch/boreas/$DAY/$DAY.tar -C $SEQ_DIR

# Rename if necessary
python $SOURCEDIR/rename_scans.py --slurm_dir $SLURM_TMPDIR/boreas

###########################
# Post-processing for Boreas data
###########################

# Separate train, validation and test
python $SOURCEDIR/separate_train_test.py --data_dir $SLURM_TMPDIR/boreas

###########################
# Test sequences for Kitti
###########################
# Prepare data for Kitti
SEQ_DIR=$SLURM_TMPDIR/Kitti
mkdir -p $SEQ_DIR
unzip ~/scratch/KITTI/data_odometry_velodyne.zip -d $SEQ_DIR
unzip ~/scratch/KITTI/data_odometry_calib.zip -d $SEQ_DIR
unzip ~/scratch/KITTI/data_odometry_labels.zip -d $SEQ_DIR

########## 2-17-2021 ########
## Start training
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#python $SOURCEDIR/train_Boreas.py --job_dir $JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas --source_dir $SOURCEDIR

## Test on Boreas dataset at epoch 150-ish
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-17_V1.0
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/boreas --source_dir $SOURCEDIR

# Create labels on 12-01
TEST_DIR=/home/hzhang34/projects/def-tbarfoot/hzhang34/KPConvAnnot/test
JOB_DIR=Log_2021-02-17_V1.0
python $SOURCEDIR/create_labels.py --job_dir $TEST_DIR/$JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas

## Test on highway-00
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-17_V1.2
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/boreas --source_dir $SOURCEDIR

## Create labels on highway-00
#TEST_DIR=/home/hzhang34/projects/def-tbarfoot/hzhang34/KPConvAnnot/test
#JOB_DIR=Log_2021-02-17_V1.2
#python $SOURCEDIR/create_labels.py --job_dir $TEST_DIR/$JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas

## Dufferin Submit on SBATCH
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-17_V1.3
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/boreas --source_dir $SOURCEDIR

## Create labels on dufferin-00
#TEST_DIR=/home/hzhang34/projects/def-tbarfoot/hzhang34/KPConvAnnot/test
#JOB_DIR=Log_2021-02-17_V1.2
#python $SOURCEDIR/create_labels.py --job_dir $TEST_DIR/$JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas

## Local full test
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-17_V1.4
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/boreas --source_dir $SOURCEDIR

## Create labels on local-00
#TEST_DIR=/home/hzhang34/projects/def-tbarfoot/hzhang34/KPConvAnnot/test
#JOB_DIR=Log_2021-02-17_V1.4
#python $SOURCEDIR/create_labels.py --job_dir $TEST_DIR/$JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas

## Kitti seq 00
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-17_V1.5
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR

## Create labels on kitti seq 00
#TEST_DIR=/home/hzhang34/projects/def-tbarfoot/hzhang34/KPConvAnnot/test
#JOB_DIR=Log_2021-02-17_V1.5
#python $SOURCEDIR/create_kitti_labels.py --job_dir $TEST_DIR/$JOB_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset

## Kitti seq 00 with an early stage trained model
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-17_V1.6
#CHKP=chkp_0100.tar
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR --chkp $CHKP

## Kitti seq 00 with reduced intensity
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-17_V1.7
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR
#
## Kitti seq 08
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V1
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-17_V1.8
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR

######### 2-23-2021 ########
## Using only point coordinates. No other features
## 45091346
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-23_V0
#FEATURE_DIM=2
#python $SOURCEDIR/train_Boreas.py --job_dir $JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas --source_dir $SOURCEDIR --feature_dims $FEATURE_DIM

## Kitti seq 00 with an early stage trained model
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-23_V0
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-23_V0.0
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR
#
## Kitti seq 08
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-23_V0
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-23_V0.1
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR
#
## Kitti seq 08 with specified chkp
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-23_V0
#LOG_DIR=$SOURCEDIR/results/Log_2021-02-23_V0.2
#python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR --chkp chkp_0050.tar
#
# Boreas
JOB_DIR=$SOURCEDIR/results/Log_2021-02-23_V0
LOG_DIR=$SOURCEDIR/results/Log_2021-02-23_V0.3
python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR --chkp chkp_0050.tar

## Include height features now
## 45091357
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-23_V1
#FEATURE_DIM=3
#python $SOURCEDIR/train_Boreas.py --job_dir $JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas --source_dir $SOURCEDIR --feature_dims $FEATURE_DIM

# Test on validation set in random order
JOB_DIR=$SOURCEDIR/results/Log_2021-02-23_V1
LOG_DIR=$SOURCEDIR/results/Log_2021-02-23_V1.0
RAND_POTENTIAL=True
python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/boreas --source_dir $SOURCEDIR --random_potential $RAND_POTENTIAL

# Test on Kitti
JOB_DIR=$SOURCEDIR/results/Log_2021-02-23_V1
LOG_DIR=$SOURCEDIR/results/Log_2021-02-23_V1.1
python $SOURCEDIR/test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --slurm_dir $SLURM_TMPDIR/Kitti/dataset --source_dir $SOURCEDIR
