#!/bin/bash

rm -rf data/

wget https://project.las.ethz.ch/static/task1b_ql4jfi6af0.zip -O task.zip -q

unzip task.zip -d data
rm task.zip