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
# On GPU 2 with batch size 8
# Fix balanced class (technically the same as before)
# Add class_w to balance loss (use inverse eq)
# Also using x, y, z but not intensity now
# Changing resolution to 0.2m
JOB_DIR=results/Log_2021-01-03_V0
python train_Buick.py --job_dir $JOB_DIR























