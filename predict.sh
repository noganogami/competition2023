#!/bin/bash

root="toyota_cars"
cars=$(ls $root)
for car in $cars; do
  if [ $(ls $root/$car/ | wc -l) -ge 400 ]; then
    python train.py --car $car --predict
  fi
done
