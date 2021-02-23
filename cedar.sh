#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request specific V100
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=150G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=5-00:00     # DD-HH:MM:SS
#SBATCH --mail-user=gareth.zhang@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python/3.7.7 cuda cudnn

SOURCEDIR=~/projects/def-tbarfoot/hzhang34/KPConvAnnot/

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt

# Prepare data
SEQ_DIR=$SLURM_TMPDIR/boreas/raw/sequences
mkdir -p $SEQ_DIR

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

# Separate train, validation and test
python $SOURCEDIR/separate_train_test.py --data_dir $SLURM_TMPDIR/boreas

######### 2-16-2021 ########
## Start training
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-16_V1
#python $SOURCEDIR/train_Boreas.py --job_dir $JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas

######### 2-17-2021 ########
## Start training
#JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V0
#python $SOURCEDIR/train_Boreas.py --job_dir $JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas

# re-start training that failed due to out of memory error
JOB_DIR=$SOURCEDIR/results/Log_2021-02-17_V2
python $SOURCEDIR/train_Boreas.py --job_dir $JOB_DIR --slurm_dir $SLURM_TMPDIR/boreas