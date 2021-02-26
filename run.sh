#!/bin/bash

######## 12-07-2020 ########
#python train_Buick.py

######## 12-08-2020 ########
#JOB_DIR=results/Log_2020-12-08_05-04-28
#LOG_DIR=results/Log_2020-12-08_05-04-28.0
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 2
#JOB_DIR=results/Log_2020-12-08_V0
#python train_Buick.py --job_dir $JOB_DIR

## On GPU 0
#JOB_DIR=results/Log_2020-12-08_V1
#python train_Buick.py --job_dir $JOB_DIR

## On GPU 1
## Evaluated at early training stage
#JOB_DIR=results/Log_2020-12-08_V0
#LOG_DIR=results/Log_2020-12-08_V0.0
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

######## 12-09-2020 ########
## On GPU 1
## Evaluated at late training stage
#JOB_DIR=results/Log_2020-12-08_V0
#LOG_DIR=results/Log_2020-12-08_V0.1
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 1
## Plot convergence
#JOB_DIR=results/Log_2020-12-08_V0
#python plot_convergence.py

## On GPU 2
## Now classes are balanced
#JOB_DIR=results/Log_2020-12-09_V0
#python train_Buick.py --job_dir $JOB_DIR

######## 12-10-2020 ########
## On GPU 1
## Evaluated at 240 epoch when classes are all balanced
#JOB_DIR=results/Log_2020-12-09_V0
#LOG_DIR=results/Log_2020-12-09_V0.0
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 1
## Evaluated at 240 epoch when classes are all balanced
## This is evaluated on test set
#JOB_DIR=results/Log_2020-12-09_V0
#LOG_DIR=results/Log_2020-12-09_V0.1
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 1
## Balanced class with validation set adding in
#JOB_DIR=results/Log_2020-12-10_V0
#python train_Buick.py --job_dir $JOB_DIR

######## 12-13-2020 ########
## On GPU 2
## Fixed the validation set problems
#JOB_DIR=results/Log_2020-12-13_V0
#python train_Buick.py --job_dir $JOB_DIR

######## 12-14-2020 ########
## Locally visualize the network performance
#JOB_DIR=results/Log_2020-12-09_V0
#KEYWORDS=EPOCH800
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## Locally visualize the network performance
#JOB_DIR=results/Log_2020-12-13_V0
#KEYWORDS=EPOCH200
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## On GPU 0
## Evaluated at 800 epoch when classes are all balanced
## This is evaluated on test set
#JOB_DIR=results/Log_2020-12-09_V0
#LOG_DIR=results/Log_2020-12-09_V0.2
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

######## 12-15-2020 ########
## On GPU 0
## Evaluated at 800 epoch when classes are all balanced
## This is evaluated on test set when changing last_min to 0.0
#JOB_DIR=results/Log_2020-12-09_V0
#LOG_DIR=results/Log_2020-12-09_V0.3
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2020-12-09_V0.3
#python plot_confusion_matrix.py --job_dir $JOB_DIR

## On GPU 0
## Without balancing dataset, how would the network perform on test set
#JOB_DIR=results/Log_2020-12-08_V0
#LOG_DIR=results/Log_2020-12-08_V0.2
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2020-12-08_V0
#KEYWORDS=EPOCH250
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## On GPU 0
## With balancing class at epoch 250, compare the performance
#JOB_DIR=results/Log_2020-12-09_V0
#LOG_DIR=results/Log_2020-12-09_V0.4
#CHKP=chkp_0250.tar
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --chkp $CHKP

## On GPU 0
## Balanced class and record validation set loss
## Also using x, y, z and intensity now
#JOB_DIR=results/Log_2020-12-15_V0
#python train_Buick.py --job_dir $JOB_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2020-12-15_V0
#KEYWORDS=EPOCH170
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

######## 12-16-2020 ########
## On GPU 0 with batch size 8
## Fix balanced class (technically the same as before)
## Add class_w to balance loss (use inverse eq)
## Also using x, y, z and intensity now
## Changing resolution to 0.2m
#JOB_DIR=results/Log_2020-12-16_V0
#python train_Buick.py --job_dir $JOB_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2020-12-16_V0
#KEYWORDS=EPOCH3
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

######## 1-3-2021 ########
## On GPU 2 with batch size 8
## Fix balanced class (technically the same as before)
## Add class_w to balance loss (use inverse eq)
## Also using x, y, z but not intensity now
## Changing resolution to 0.2m
#JOB_DIR=results/Log_2021-01-03_V0
#python train_Buick.py --job_dir $JOB_DIR

######## 1-4-2021 ########
## On GPU 2 with batch size 8
## Everything same with the previous job
## BUT now with utias section excluded with hard indices
#JOB_DIR=results/Log_2021-01-04_V0
#python train_Buick.py --job_dir $JOB_DIR

## On GPU 1 with batch size 2
## Everything same with the previous job
## BUT now with utias section excluded with hard indices
#JOB_DIR=results/Log_2021-01-04_V1
#python train_Buick.py --job_dir $JOB_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-01-04_V0
#KEYWORDS=EPOCH250
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-01-04_V1
#KEYWORDS=EPOCH3
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## On GPU 0
#JOB_DIR=results/Log_2021-01-04_V0
#LOG_DIR=results/Log_2021-01-04_V0.0
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 0
## This is tested on SemanticKitti dataset
#JOB_DIR=results/Log_2021-01-04_V0
#LOG_DIR=results/Log_2021-01-04_V0.1
#python test_SemanticKitti.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 0 with batch size 2
## Just want to see if SemanticKitti dataset works
##python train_SemanticKitti.py
## Locally visualize the network performance
#JOB_DIR=results/Log_2020-12-16_V0
#KEYWORDS=EPOCH3
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## On GPU 0
#JOB_DIR=results/Log_2021-01-04_V0
#LOG_DIR=results/Log_2021-01-04_V0.2
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-01-04_V0.2
#python plot_confusion_matrix.py --job_dir $JOB_DIR

## On GPU 3
#JOB_DIR=results/Log_2021-01-04_V0
#LOG_DIR=results/Log_2021-01-04_V0.3
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-01-04_V0.3
#python plot_confusion_matrix.py --job_dir $JOB_DIR

## Sample the annotations for visualizations
#python visualize_annotations.py

######## 1-6-2021 ########
## On GPU 3
#JOB_DIR=results/Log_2021-01-04_V0
#LOG_DIR=results/Log_2021-01-04_V0.4
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 3
## Test on Buick dataset
#JOB_DIR=results/Log_2021-01-04_V0
#LOG_DIR=results/Log_2021-01-04_V0.5
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 3
## Test on Buick dataset
#JOB_DIR=results/Log_2020-12-09_V0
#LOG_DIR=results/Log_2020-12-09_V0.5
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 3
## This is tested on SemanticKitti dataset
#JOB_DIR=results/Log_2021-01-04_V0
#LOG_DIR=results/Log_2021-01-04_V0.6
#python test_SemanticKitti.py --job_dir $JOB_DIR --log_dir $LOG_DIR

######## 1-9-2021 ########
## On GPU 1
## Test on Buick dataset
#JOB_DIR=results/Log_2021-01-04_V0
#LOG_DIR=results/Log_2021-01-04_V0.7
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

######### 1-12-2021 ########
## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-01-04_V0.7
#python plot_confusion_matrix.py --job_dir $JOB_DIR

######### 1-15-2021 ########
## On GPU 3 with batch size 2
## Use height and intensity
#JOB_DIR=results/Log_2021-01-15_V0
#python train_Boreas.py --job_dir $JOB_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-01-15_V0
#KEYWORDS=EPOCH30
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

######### 1-17-2021 ########
## On GPU 1
## Test on Boreas dataset
#JOB_DIR=results/Log_2021-01-15_V0
#LOG_DIR=results/Log_2021-01-15_V0.0
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-01-15_V0.0
#python plot_confusion_matrix.py --job_dir $JOB_DIR

## On GPU 1
## Test on Buick dataset
#JOB_DIR=results/Log_2021-01-15_V0
#LOG_DIR=results/Log_2021-01-15_V0.1
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-01-15_V0.1
#python plot_confusion_matrix.py --job_dir $JOB_DIR

######### 1-18-2021 ########
## On GPU 1
## Test on Boreas dataset sequentially
#JOB_DIR=results/Log_2021-01-15_V0
#LOG_DIR=results/Log_2021-01-15_V0.2
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## On GPU 3 with batch size 8
## Now the potential is updated
## Use height and intensity
#JOB_DIR=results/Log_2021-01-18_V0
#python train_Boreas.py --job_dir $JOB_DIR

## On GPU 3
## Test on Boreas dataset sequentially
#JOB_DIR=results/Log_2021-01-15_V0
#LOG_DIR=results/Log_2021-01-15_V0.3
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Test on Boreas dataset sequentially
## Write every predicted .ply label to file
#JOB_DIR=results/Log_2021-01-15_V0
#LOG_DIR=results/Log_2021-01-15_V0.4
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Test on Boreas dataset sequentially
## Write every predicted .ply label to file
#JOB_DIR=results/Log_2021-01-15_V0
#LOG_DIR=results/Log_2021-01-15_V0.5
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

#python create_labels.py

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-01-15_V0.5
#python plot_confusion_matrix.py --job_dir $JOB_DIR

######### 2-6-2021 ########
#python separate_train_test.py

## On GPU 1
## Test seen and unseen dataset
#JOB_DIR=results/Log_2021-02-06_V0
#python train_Boreas.py --job_dir $JOB_DIR

## On GPU 3
## Test on Boreas dataset at epoch 15-ish
#JOB_DIR=results/Log_2021-02-06_V0
#LOG_DIR=results/Log_2021-02-06_V0.0
#RAND_POTENTIAL=True
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --random_potential $RAND_POTENTIAL

## On GPU 1
## Using 5 feature dim, remove ignored labels, enlarge radius to 150
#JOB_DIR=results/Log_2021-02-06_V1
#python train_Boreas.py --job_dir $JOB_DIR

## On GPU 1
## Using 5 feature dim, enlarge radius to 150
## throw away 12-01
#JOB_DIR=results/Log_2021-02-06_V2
#python train_Boreas.py --job_dir $JOB_DIR

## On GPU 1
## Re-run
## throw away 12-01
#JOB_DIR=results/Log_2021-02-08_V0
#python train_Boreas.py --job_dir $JOB_DIR

## On GPU 1
## Re-run but with 12-01 added back
#JOB_DIR=results/Log_2021-02-08_V1
#python train_Boreas.py --job_dir $JOB_DIR

## On GPU 1
## Re-run but with 12-01 added back
#JOB_DIR=results/Log_2021-02-08_V2
#python train_Boreas.py --job_dir $JOB_DIR

## On Cedar
#JOB_DIR=results/Log_2021-02-08_V3
#python train_Boreas.py --job_dir $JOB_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-02-08_V3
#KEYWORDS=EPOCH11
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## On GPU 3
## Test on Boreas highway-00 dataset at epoch 15-ish
#JOB_DIR=results/Log_2021-02-08_V3
#LOG_DIR=results/Log_2021-02-08_V3.0
#RAND_POTENTIAL=False
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Create labels on highway-00
#TEST_DIR=/home/gzh/Documents/research/src/KPConvAnnot/test
#JOB_DIR=Log_2021-02-08_V3.0
#python create_labels.py --job_dir $TEST_DIR/$JOB_DIR

## On GPU 0
#JOB_DIR=results/Log_2021-02-08_V4
#python train_Boreas.py --job_dir $JOB_DIR

## On GPU 3
## Test on Boreas dataset at epoch 15-ish
#JOB_DIR=results/Log_2021-02-08_V3
#LOG_DIR=results/Log_2021-02-08_V3.1
#RAND_POTENTIAL=True
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR --random_potential $RAND_POTENTIAL

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-02-08_V3.1
#python plot_confusion_matrix.py --job_dir $JOB_DIR

#python separate_train_test.py

######### 2-16-2021 ########
## On GPU 3
## Experiment with new directory architecture setup
#JOB_DIR=results/Log_2021-02-16_V0
#python train_Boreas.py --job_dir $JOB_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-02-16_V0
#KEYWORDS=EPOCH11
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-02-16_V1
#KEYWORDS=EPOCH11
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## On GPU 3
## Test on Boreas dataset at epoch 150-ish
#JOB_DIR=results/Log_2021-02-16_V0
#LOG_DIR=results/Log_2021-02-16_V0.0
#RAND_POTENTIAL=True
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-02-16_V0.0
#python plot_confusion_matrix.py --job_dir $JOB_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-02-17_V0
#KEYWORDS=EPOCH11
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## On GPU 3
## Test on Boreas dataset at epoch 150-ish
#JOB_DIR=results/Log_2021-02-16_V0
#LOG_DIR=results/Log_2021-02-16_V0.1
#RAND_POTENTIAL=True
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-02-16_V0.1
#FNAME=boreas-2020-12-01-13-26_0000333.ply
#python plot_confusion_matrix_per_frame.py --job_dir $JOB_DIR --fname $FNAME

## Plot confusion matrices for reports
#JOB_DIR=test/Log_2021-02-16_V0.1
#python plot_confusion_matrix.py --job_dir $JOB_DIR

## On GPU 3
## Test on Boreas dataset at epoch 150-ish
#JOB_DIR=results/Log_2021-02-16_V0
#LOG_DIR=results/Log_2021-02-16_V0.1
#RAND_POTENTIAL=True
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-02-17_V1
#KEYWORDS=EPOCH11
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-02-17_V2
#KEYWORDS=EPOCH11
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## On GPU 0
## Test on Boreas dataset at epoch 150-ish
#JOB_DIR=results/Log_2021-02-17_V1
#LOG_DIR=results/Log_2021-02-17_V1.0
#python test_models.py --job_dir $JOB_DIR --log_dir $LOG_DIR

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-02-23_V0
#KEYWORDS=EPOCH11
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

# Locally visualize the network performance
JOB_DIR=results/Log_2021-02-23_V1
KEYWORDS=EPOCH11
python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS

## Locally visualize the network performance
#JOB_DIR=results/Log_2021-02-23_V2
#KEYWORDS=EPOCH11
#python visualize_performance.py --job_dir $JOB_DIR --keywords $KEYWORDS