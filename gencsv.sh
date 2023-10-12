#!/bin/bash

root="toyota_cars"
cars=$(ls $root)
for car in $cars; do
  if [ $(ls $root/$car/ | wc -l) -ge 400 ]; then
    touch $car.csv
    echo file_name, > $car.csv
  fi
done
