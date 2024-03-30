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
pip install numpy matplotlib pandas
```

## Run PySpark
- Run main script to fetch all recommendations
```
bash run.sh
```
- Run script to fetch recommendations by popularity
```
bash popular_movies.sh
```
- Run script to fetch recommendations by matrix factorization
```
bash matrix_factorization.sh
```