import os
from typing import Optional

import numpy as np

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

class MovieLens20m:
    def __init__(self, spark, data_dir: str = "ml-20m") -> None:
        self.spark = spark
        self.movie_fname = os.path.join(data_dir, "movies.csv")
        self.ratings_fname = os.path.join(data_dir, "ratings.csv")

        # Load datasets
        self.movies_df = self.spark.read.option("header", True).csv(self.movie_fname)
        self.ratings_df = self.spark.read.option("header", True).csv(self.ratings_fname)

        # Compute movie popularity and rating statistics
        self.popular_movies_df = self.compute_movie_popularity()
        self.n_rating_statistics = self.compute_statistics_for_number_of_rating()

    def get_movies_df(self):
        return self.movies_df
    
    def get_ratings_df(self):
        return self.ratings_df

    def compute_movie_popularity(self):
        popular_movies_df = self.ratings_df.groupBy("movieId")\
            .agg(
                F.count("userId").alias("n_rating"), 
                F.avg("rating").alias("avg_rating"), 
                F.stddev("rating").alias("std_rating")
            )
        popular_movies_df = popular_movies_df.join(
            self.movies_df, popular_movies_df.movieId == self.movies_df.movieId
        )
        return popular_movies_df

    def compute_statistics_for_number_of_rating(self):
        n_rating_stats = self.popular_movies_df.select([
            F.mean('n_rating').alias("n_rating_avg"), 
            F.min('n_rating').alias("n_rating_min"), 
            F.max('n_rating').alias("n_rating_max")
        ]).collect()
        if len(n_rating_stats) == 0:
            raise "Not Enough Data"
        
        n_rating_stats = n_rating_stats[0]
        n_rating_avg = n_rating_stats.n_rating_avg
        n_rating_min = n_rating_stats.n_rating_min
        n_rating_max = n_rating_stats.n_rating_max
        percentiles = self.popular_movies_df.approxQuantile('n_rating', [0.25, 0.50, 0.75], 0)
        n_rating_25_percentile, n_rating_50_percentile, n_rating_75_percentile = percentiles
        return {
            "n_rating_avg": n_rating_avg,
            "n_rating_min": n_rating_min,
            "n_rating_max": n_rating_max,
            "n_rating_25_percentile": n_rating_25_percentile,
            "n_rating_50_percentile": n_rating_50_percentile,
            "n_rating_75_percentile": n_rating_75_percentile,
        }
    
    def get_statistics_for_number_of_rating(self):
        return self.n_rating_statistics
    
    def rank_movies_by_number_of_ratings(self, min_n_rating: Optional[int] = 0):
        popular_movies_df = self.filter_movies_with_min_n_rating(min_n_rating)
        popular_movies_by_number_of_ratings_df = popular_movies_df.sort(F.desc("n_rating"))
        return popular_movies_by_number_of_ratings_df

    def rank_movies_by_average_ratings(self, min_n_rating: Optional[int] = None):
        popular_movies_df = self.filter_movies_with_min_n_rating(min_n_rating)
        popular_movies_by_average_ratings_df = popular_movies_df.sort(F.desc("avg_rating"))
        return popular_movies_by_average_ratings_df

    def rank_movies_by_std_ratings(self, min_n_rating: Optional[int] = None):
        popular_movies_df = self.filter_movies_with_min_n_rating(min_n_rating)
        popular_movies_by_std_ratings_df = popular_movies_df\
            .filter(self.popular_movies_df.std_rating != np.nan)\
            .sort(F.desc("std_rating"))
        return popular_movies_by_std_ratings_df
    
    def filter_movies_with_min_n_rating(self, min_n_rating: Optional[int] = 0):
        if min_n_rating == 0:
            return self.popular_movies_df
        else:
            return self.popular_movies_df.filter(self.popular_movies_df.n_rating >= min_n_rating)

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Project").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)

    # Compute rating statistics to understand the skew in the number of ratings
    n_rating_statistics = movielens20m.get_statistics_for_number_of_rating()
    print("n_rating_statistics:", n_rating_statistics)

    # Rank popular movies by the number of ratings
    movielens20m.rank_movies_by_number_of_ratings(
        min_n_rating=n_rating_statistics["n_rating_75_percentile"]
    ).show(10)

    # Rank popular movies by the average ratings
    movielens20m.rank_movies_by_average_ratings(
        min_n_rating=n_rating_statistics["n_rating_75_percentile"]
    ).show(10)

    # Rank popular movies that has polarized ratings
    movielens20m.rank_movies_by_std_ratings(
        min_n_rating=n_rating_statistics["n_rating_75_percentile"]
    ).show(10)
