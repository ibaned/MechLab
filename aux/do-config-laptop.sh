#!/bin/bash -ex

# Modify these paths for your system
GOAL_DIR=/home/bng/codes/goal/install

cmake .. \
-DCMAKE_CXX_COMPILER="mpicxx" \
-DCMAKE_INSTALL_PREFIX=../install \
-DBUILD_TESTING=ON \
-DGoal_PREFIX=$GOAL_DIR \
2>&1 | tee config_log
