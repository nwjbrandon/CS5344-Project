import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

class MovieLens20m:
    def __init__(self, spark, data_dir: str = "ml-20m") -> None:
        self.spark = spark
        self.movie_fname = os.path.join(data_dir, "movies.csv")
        self.ratings_fname = os.path.join(data_dir, "ratings.csv")

        self.movies_df = self.spark.read.option("header", True).csv(self.movie_fname)
        self.ratings_df = self.spark.read.option("header", True).csv(self.ratings_fname)
        self.popular_movies_df = self.compute_movie_popularity()

    def get_movies_df(self):
        return self.movies_df
    
    def get_ratings_df(self):
        return self.ratings_df

    def compute_movie_popularity(self):
        popular_movies_df = self.ratings_df.groupBy("movieId").agg(F.count("userId"), F.avg("rating")).withColumnRenamed("count(userId)", "n_ratings").withColumnRenamed("avg(rating)", "avg_rating")
        popular_movies_df = popular_movies_df.join(self.movies_df, popular_movies_df.movieId == self.movies_df.movieId)
        return popular_movies_df
    
    def get_number_of_rating_statistics(self):
        return self.popular_movies_df.select([F.mean('n_ratings'), F.min('n_ratings'), F.max('n_ratings')])

    def get_average_rating_statistics(self):
        return self.popular_movies_df.select([F.mean('avg_rating'), F.min('avg_rating'), F.max('avg_rating')])

    def rank_movies_by_number_of_ratings(self):
        popular_movies_by_number_of_ratings_df = self.popular_movies_df.sort(F.desc("n_ratings"))
        return popular_movies_by_number_of_ratings_df

    def rank_movies_by_average_ratings(self):
        popular_movies_by_average_ratings_df = self.popular_movies_df.sort(F.desc("avg_rating"))
        return popular_movies_by_average_ratings_df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Lab2").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    movielens20m.get_number_of_rating_statistics().show(10)
    movielens20m.get_average_rating_statistics().show(10)
    movielens20m.rank_movies_by_number_of_ratings().show(10)
    movielens20m.rank_movies_by_average_ratings().show(10)
