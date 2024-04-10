import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from dataset import MovieLens20m, MovieLens10m


class PopularMovies:
    def __init__(self, movielens20m: MovieLens20m) -> None:
        self.movielens20m = movielens20m
        self.ratings_df = self.movielens20m.get_ratings_df()
        self.movies_df = self.movielens20m.get_movies_df()
        self.features = self.movielens20m.extract_features()

    def rank_by_number_of_rating_per_movie(self, min_number_of_rating_per_movie_threshold: int = 0):
        df = self.movies_df.join(self.features["ratings_per_movie_df"], ["movieID"])
        df = df.select(["movieId", "title", "genres", "number_of_rating_per_movie"]).drop_duplicates()
        df = self.filter_movie_with_at_least_k_number_of_rating(df, min_number_of_rating_per_movie_threshold)
        df = df.sort(F.desc("number_of_rating_per_movie"))
        return df

    def rank_by_average_rating_per_movie(self, min_number_of_rating_per_movie_threshold: int = 0):
        df = self.movies_df.join(self.features["ratings_per_movie_df"], ["movieID"])
        df = df.select(["movieId", "title", "genres", "average_rating_per_movie", "number_of_rating_per_movie"]).drop_duplicates()
        df = self.filter_movie_with_at_least_k_number_of_rating(df, min_number_of_rating_per_movie_threshold)
        df = df.sort(F.desc("average_rating_per_movie"))
        df = df.select(["movieId", "title", "genres", "average_rating_per_movie"])
        return df

    def rank_by_standard_deviation_rating_per_movie(self, min_number_of_rating_per_movie_threshold: int = 0):
        df = self.movies_df.join(self.features["ratings_per_movie_df"], ["movieID"])
        df = df.select(["movieId", "title", "genres", "standard_deviation_rating_per_movie", "number_of_rating_per_movie"]).drop_duplicates()
        df = self.filter_movie_with_at_least_k_number_of_rating(df, min_number_of_rating_per_movie_threshold)
        df = df.filter(df["standard_deviation_rating_per_movie"] != np.nan).sort(F.desc("standard_deviation_rating_per_movie"))
        df = df.select(["movieId", "title", "genres", "standard_deviation_rating_per_movie"])
        return df

    def filter_movie_with_at_least_k_number_of_rating(self, df, k: int = 0):
        return df.filter(df["number_of_rating_per_movie"] >= k)


def recommend_movies_by_popularity(movielens20m: MovieLens20m):
    popular_movies = PopularMovies(movielens20m)

    df = popular_movies.rank_by_number_of_rating_per_movie(min_number_of_rating_per_movie_threshold=200)
    df.show(10)

    df = popular_movies.rank_by_average_rating_per_movie(min_number_of_rating_per_movie_threshold=200)
    df = df.show(10)

    df = popular_movies.rank_by_standard_deviation_rating_per_movie(min_number_of_rating_per_movie_threshold=200)
    df = df.show(10)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project Popular Movies").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    recommend_movies_by_popularity(movielens20m)
