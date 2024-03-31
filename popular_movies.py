from typing import Optional

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from dataset import MovieLens20m


class PopularMovies:
    def __init__(self, movielens20m: MovieLens20m) -> None:
        self.movielens20m = movielens20m

        self.popular_movies_df = self.compute_popular_movies()

    def compute_popular_movies(self):
        ratings_df = self.movielens20m.get_ratings_df()
        movies_df = self.movielens20m.get_movies_df()

        popular_movies_df = ratings_df.groupBy("movieId").agg(F.count("userId").alias("n_rating"), F.avg("rating").alias("avg_rating"), F.stddev("rating").alias("std_rating"))
        popular_movies_df = popular_movies_df.join(movies_df, ["movieId"])
        return popular_movies_df

    def get_popular_movies_df(self):
        return self.popular_movies_df

    def rank_movies_by_number_of_rating(self, min_n_rating_threshold: Optional[int] = 0):
        popular_movies_df = self.preprocess_movies_df(min_n_rating_threshold)
        popular_movies_by_number_of_ratings_df = popular_movies_df.sort(F.desc("n_rating"))
        return popular_movies_by_number_of_ratings_df

    def rank_movies_by_average_rating(self, min_n_rating_threshold: Optional[int] = None):
        popular_movies_df = self.preprocess_movies_df(min_n_rating_threshold)
        popular_movies_by_average_ratings_df = popular_movies_df.sort(F.desc("avg_rating"))
        return popular_movies_by_average_ratings_df

    def rank_movies_by_std_rating(self, min_n_rating_threshold: Optional[int] = None):
        popular_movies_df = self.preprocess_movies_df(min_n_rating_threshold)
        popular_movies_by_std_ratings_df = popular_movies_df.filter(self.popular_movies_df.std_rating != np.nan).sort(F.desc("std_rating"))
        return popular_movies_by_std_ratings_df

    def preprocess_movies_df(self, min_n_rating_threshold: Optional[int] = 0):
        if min_n_rating_threshold == 0:
            return self.popular_movies_df
        else:
            return self.popular_movies_df.filter(self.popular_movies_df.n_rating >= min_n_rating_threshold)


def recommend_movies_by_popularity(movielens20m: MovieLens20m):
    popular_movies = PopularMovies(movielens20m)

    # Rank popular movies by the number of ratings
    popular_movies.rank_movies_by_number_of_rating(min_n_rating_threshold=200).show(10)

    # Rank popular movies by the average ratings
    popular_movies.rank_movies_by_average_rating(min_n_rating_threshold=200).show(10)

    # Rank popular movies that has polarized ratings
    popular_movies.rank_movies_by_std_rating(min_n_rating_threshold=200).show(10)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project Popular Movies").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    recommend_movies_by_popularity(movielens20m)
