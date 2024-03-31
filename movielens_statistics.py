from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from dataset import MovieLens20m
from popular_movies import PopularMovies


class MovieLensStatistics:
    def __init__(self, movielens20m: MovieLens20m) -> None:
        self.movielens20m = movielens20m
        self.popular_movies = PopularMovies(movielens20m)
        self.popular_movies_df = self.popular_movies.get_popular_movies_df()

    def visualise_movie_n_rating_boxplot(self):
        percentiles = self.popular_movies_df.approxQuantile("n_rating", [0.00, 0.25, 0.50, 0.75, 1.00], 0)
        whislo = percentiles[0]
        q1 = percentiles[1]
        med = percentiles[2]
        q3 = percentiles[3]
        whishi = percentiles[4]
        mean = self.compute_average_n_rating()

        fig, ax = plt.subplots()
        boxes = [
            {
                "label": "Number Of Rating",
                "whislo": np.log(whislo),
                "q1": np.log(q1),
                "med": np.log(med),
                "q3": np.log(q3),
                "whishi": np.log(whishi),
                "fliers": [],
            }
        ]
        ax.bxp(boxes, showfliers=False)
        ax.set_ylabel("Logarithm Of Number Of Rating Per Movie")
        fig.suptitle("Box Plot Of Log Distribution Of Number Of Rating Per Movie")
        plt.savefig("imgs/boxplot_distribution_of_number_of_rating.png")
        plt.close()
        return {
            "whislo": whislo,
            "q1": q1,
            "med": med,
            "q3": q3,
            "whishi": whishi,
            "mean": mean,
        }

    def visualise_n_rating_per_user_boxplot(self):
        ratings_df = self.movielens20m.get_ratings_df()
        user_ratings_df = ratings_df.groupBy("userId").count()
        percentiles = user_ratings_df.approxQuantile("count", [0.00, 0.25, 0.50, 0.75, 1.00], 0)
        whislo = percentiles[0]
        q1 = percentiles[1]
        med = percentiles[2]
        q3 = percentiles[3]
        whishi = percentiles[4]
        mean = self.compute_average_n_rating_per_user()

        fig, ax = plt.subplots()
        boxes = [
            {
                "label": "Number Of Rating",
                "whislo": np.log(whislo),
                "q1": np.log(q1),
                "med": np.log(med),
                "q3": np.log(q3),
                "whishi": np.log(whishi),
                "fliers": [],
            }
        ]
        ax.bxp(boxes, showfliers=False)
        ax.set_ylabel("Logarithm Of Number Of Rating Per User")
        fig.suptitle("Box Plot Of Log Number Of Rating Per User")
        plt.savefig("imgs/boxplot_log_number_of_rating_per_user.png")
        plt.close()
        return {
            "whislo": whislo,
            "q1": q1,
            "med": med,
            "q3": q3,
            "whishi": whishi,
            "mean": mean,
        }

    def compute_average_n_rating(self):
        n_rating_stats = self.popular_movies_df.select(
            [
                F.mean("n_rating").alias("n_rating_avg"),
            ]
        ).collect()

        return n_rating_stats[0].n_rating_avg

    def compute_average_n_rating_per_user(self):
        ratings_df = self.movielens20m.get_ratings_df()
        user_ratings_df = ratings_df.groupBy("userId").count()
        user_ratings_stats = user_ratings_df.select(
            [
                F.mean("count").alias("count_avg"),
            ]
        ).collect()

        return user_ratings_stats[0].count_avg

    def rank_movies_by_number_of_rating(self, min_n_rating_threshold: Optional[int] = 0):
        popular_movies_df = self.filter_movies_with_at_least_n_rating(min_n_rating_threshold)
        popular_movies_by_number_of_ratings_df = popular_movies_df.sort(F.desc("n_rating"))
        return popular_movies_by_number_of_ratings_df

    def rank_movies_by_average_rating(self, min_n_rating_threshold: Optional[int] = None):
        popular_movies_df = self.filter_movies_with_at_least_n_rating(min_n_rating_threshold)
        popular_movies_by_average_ratings_df = popular_movies_df.sort(F.desc("avg_rating"))
        return popular_movies_by_average_ratings_df

    def rank_movies_by_std_rating(self, min_n_rating_threshold: Optional[int] = None):
        popular_movies_df = self.filter_movies_with_at_least_n_rating(min_n_rating_threshold)
        popular_movies_by_std_ratings_df = popular_movies_df.filter(self.popular_movies_df.std_rating != np.nan).sort(F.desc("std_rating"))
        return popular_movies_by_std_ratings_df

    def filter_movies_with_at_least_n_rating(self, min_n_rating_threshold: Optional[int] = 0):
        if min_n_rating_threshold == 0:
            return self.popular_movies_df
        else:
            return self.popular_movies_df.filter(self.popular_movies_df.n_rating >= min_n_rating_threshold)

    def get_timespan_of_ratings(self):
        timestamps = (
            self.movielens20m.get_ratings_df()
            .select(
                [
                    F.min("timestamp").alias("min_timestamp"),
                    F.max("timestamp").alias("max_timestamp"),
                ]
            )
            .collect()
        )

        min_timestamp = timestamps[0].min_timestamp
        max_timestamp = timestamps[0].max_timestamp

        timespan_in_days = (max_timestamp - min_timestamp) / 60 / 60 / 24
        timespan_in_years = timespan_in_days / 365.25

        return {
            "timespan_in_days": timespan_in_days,
            "timespan_in_years": timespan_in_years,
        }

    def get_dataset_sizes(self):
        n_ratings = self.movielens20m.get_ratings_df().count()
        n_movies = self.movielens20m.get_movies_df().count()

        return {
            "n_ratings": n_ratings,
            "n_movies": n_movies,
        }

    def visualise_genres_barplot(self):
        movies_df = self.movielens20m.get_movies_df()
        genres_df = movies_df.select(F.col("genres"))
        genres_df = genres_df.select(F.explode(F.split(F.col("genres"), "\\|")).alias("genres"))
        genres_df = genres_df.groupBy("genres").count()
        n_genres = genres_df.count() - 1

        genres_df = genres_df.toPandas()
        genres_distribution = genres_df.to_dict("records")
        genres_df = genres_df.set_index("genres")
        genres_df.plot.bar()

        plt.title("Bar Plot Of Genres Distribution In MovieLens20m")
        plt.ylabel("Number Of Movies")
        plt.savefig("imgs/genres_distribution.png", bbox_inches="tight")
        plt.close()

        return {"n_genres": n_genres, "distribution": genres_distribution}

    def visualise_genres_rating_barplot(self):
        genres_rating_df = self.popular_movies_df.select(["avg_rating", "genres"])

        genres_df = genres_rating_df.select("avg_rating", F.explode(F.split(F.col("genres"), "\\|")).alias("genres"))
        genres_df = genres_df.groupBy("genres").agg(F.avg("avg_rating").alias("rating"))

        genres_df = genres_df.toPandas()
        genres_df = genres_df.set_index("genres")
        genres_df.plot.bar()

        plt.title("Bar Plot Of Ratings Of Genres Distribution In MovieLens20m")
        plt.ylabel("Average Rating")
        plt.savefig("imgs/genres_rating_distribution.png", bbox_inches="tight")
        plt.close()

    def visualise_genres_barplot_for_user(self, user_id=5):
        movies_df = self.movielens20m.get_movies_df()
        ratings_df = self.movielens20m.get_ratings_df()
        user_ratings_df = ratings_df.filter(ratings_df.userId == user_id)

        user_movies_df = user_ratings_df.join(movies_df, ["movieId"])

        genres_df = user_movies_df.select(F.col("genres"))
        genres_df = genres_df.select(F.explode(F.split(F.col("genres"), "\\|")).alias("genres"))
        genres_df = genres_df.groupBy("genres").count()

        genres_df = genres_df.toPandas()
        genres_df = genres_df.set_index("genres")
        genres_df.plot.bar()

        plt.title(f"Bar Plot Of Genres Distribution In MovieLens20m for User {user_id}")
        plt.ylabel("Number Of Movies")
        plt.savefig(f"imgs/genres_distribution_for_user_{user_id}.png", bbox_inches="tight")
        plt.close()

    def visualise_n_rating_and_avg_rating_scatterplot(self, current_year=2024):
        df = self.popular_movies_df.withColumn("log_n_rating", F.log10(F.col("n_rating")))
        df = df.withColumn("released_year", F.regexp_extract(F.col("title"), r"(?<=\()(\d+)(?=\))", 1).cast(IntegerType()))
        df = df.withColumn("movie_age", current_year - df["released_year"])
        df = df.filter(df.movie_age != np.nan)

        # TODO: Someone help me fixed the 3 rows extracts the wrong year
        df = df.filter(df.movie_age < 200)

        df = df.select(["movieId", "title", "n_rating", "movie_age"])

        df = df.toPandas()
        plt.scatter(x=df["movie_age"], y=df["n_rating"], s=2)
        plt.title("Scatter Plot Of Number Of Rating And Movie Age")
        plt.xlabel("Movie Age (years)")
        plt.ylabel("Number Of Ratings")
        plt.savefig("imgs/scatterplot_of_n_rating_and_movie_age.png", bbox_inches="tight")
        plt.close()


def compute_movielens_statistics(movielens20m: MovieLens20m):
    movielens_statistics = MovieLensStatistics(movielens20m)

    dataset_sizes = movielens_statistics.get_dataset_sizes()
    print("dataset_sizes: ", dataset_sizes)

    timespan = movielens_statistics.get_timespan_of_ratings()
    print("timespan:", timespan)

    rating_statistics = movielens_statistics.visualise_movie_n_rating_boxplot()
    print("rating_statistics: ", rating_statistics)

    user_rating_statistics = movielens_statistics.visualise_n_rating_per_user_boxplot()
    print("user_rating_statistics: ", user_rating_statistics)

    genres_statistics = movielens_statistics.visualise_genres_barplot()
    print("genres_statistics: ", genres_statistics)

    movielens_statistics.visualise_genres_rating_barplot()
    movielens_statistics.visualise_n_rating_and_avg_rating_scatterplot()

    movielens_statistics.rank_movies_by_number_of_rating(min_n_rating_threshold=50).show(10)
    movielens_statistics.rank_movies_by_average_rating(min_n_rating_threshold=50).show(10)
    movielens_statistics.rank_movies_by_std_rating(min_n_rating_threshold=50).show(10)

    movielens_statistics.visualise_genres_barplot_for_user(user_id=3)
    movielens_statistics.visualise_genres_barplot_for_user(user_id=5)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    compute_movielens_statistics(movielens20m)
