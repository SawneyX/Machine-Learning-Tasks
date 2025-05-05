#!/bin/bash

rm -rf data/

wget https://project.las.ethz.ch/static/task1a_do4bq81me.zip -O task.zip -q

unzip task.zip -d data
rm task.zip