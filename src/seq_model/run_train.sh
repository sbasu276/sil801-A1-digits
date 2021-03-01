#!/bin/bash

for i in {0..4}
do
	python train.py --set=$i
done
