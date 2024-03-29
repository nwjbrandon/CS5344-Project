# CS5344 Project

## Dataset
- Download MovieLens 20M from this [link](https://grouplens.org/datasets/movielens/20m/)
- Unzip the file `ml-20m.zip` into the base directory of this repository

## Install dependencies
- Install docker (Refer to Dockerfile for individual dependencies)
- Build docker image and enter docker container via bash
```
bash env.sh
```
- Install additional dependencies inside docker container bash shell
```
pip install numpy
```

## Run PySpark
- Run application
```
bash run.sh
```