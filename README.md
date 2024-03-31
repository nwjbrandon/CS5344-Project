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
- Run script to visualise rating and movie statistics
```
bash movielens_statistics.sh
```

## MovieLens Data Exploration
- Log distribution of number of rating for each movie shows the number of rating is highly skewed

<center>

Percentile | Value
--- | ---
Whislo | 1
Q1 | 3
Med | 18
Q3 | 205
Whishi | 67310

<img src="imgs/boxplot_distribution_pf_number_of_rating.png">

</center>

## Issues
- Solve java.lang.OutOfMemoryError: Java heap space (https://stackoverflow.com/questions/50842877/java-lang-outofmemoryerror-java-heap-space-using-docker)
