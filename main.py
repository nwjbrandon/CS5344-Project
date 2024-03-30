from pyspark.sql import SparkSession

from dataset import MovieLens20m
from matrix_factorization import recommend_movies_by_matrix_factorization
from popular_movies import recommend_movies_by_popularity

if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)

    recommend_movies_by_popularity(movielens20m)
    recommend_movies_by_matrix_factorization(movielens20m, threshold=3.5)
