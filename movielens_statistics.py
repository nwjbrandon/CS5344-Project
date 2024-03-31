from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

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
        ax.set_ylabel("Logarithm Of Number Of Rating")
        fig.suptitle("Box Plot Of Log Distribution Of Number Of Rating")
        plt.savefig("imgs/boxplot_distribution_pf_number_of_rating.png")
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


def compute_movielens_statistics(movielens20m: MovieLens20m):
    movielens_statistics = MovieLensStatistics(movielens20m)

    dataset_sizes = movielens_statistics.get_dataset_sizes()
    print("dataset_sizes: ", dataset_sizes)

    timespan = movielens_statistics.get_timespan_of_ratings()
    print("timespan:", timespan)

    n_rating_statistics = movielens_statistics.visualise_movie_n_rating_boxplot()
    print("n_rating_statistics: ", n_rating_statistics)

    movielens_statistics.rank_movies_by_number_of_rating(min_n_rating_threshold=50).show(10)
    movielens_statistics.rank_movies_by_average_rating(min_n_rating_threshold=50).show(10)
    movielens_statistics.rank_movies_by_std_rating(min_n_rating_threshold=50).show(10)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    compute_movielens_statistics(movielens20m)
