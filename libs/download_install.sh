#! /usr/bin/env bash
LIB_DIR=./tdIF #$(pwd)

# # Download YOLOX
# cd $LIB_DIR/libs
# git clone --no-checkout git@github.com:Megvii-BaseDetection/YOLOX.git
# cd $LIB_DIR/libs/YOLOX
# git checkout 618fd8c08b2bc5fac9ffbb19a3b7e039ea0d5b9a

pip install -e $LIB_DIR/libs/YOLOX
# pip install seaborn
