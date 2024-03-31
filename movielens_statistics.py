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

        fig, ax = plt.subplots()
        boxes = [
            {
                "label": "Number Of Rating",
                "whislo": np.log(whislo),  # Bottom whisker position
                "q1": np.log(q1),  # First quartile (25th percentile)
                "med": np.log(med),  # Median         (50th percentile)
                "q3": np.log(q3),  # Third quartile (75th percentile)
                "whishi": np.log(whishi),  # Top whisker position
                "fliers": [],  # Outliers
            }
        ]
        ax.bxp(boxes, showfliers=False)
        ax.set_ylabel("Logarithm Of Number Of Rating")
        fig.suptitle("Box Plot Of Log Distribution Of Number Of Rating")
        plt.savefig("imgs/boxplot_distribution_pf_number_of_rating.png")
        plt.close()
        return percentiles

    # def compute_rating_statistics(self):
    #     n_rating_stats = self.popular_movies_df.select(
    #         [
    #             F.mean("n_rating").alias("n_rating_avg"),
    #             F.min("n_rating").alias("n_rating_min"),
    #             F.max("n_rating").alias("n_rating_max"),
    #         ]
    #     ).collect()
    #     if len(n_rating_stats) == 0:
    #         raise "Not Enough Data"

    #     n_rating_stats = n_rating_stats[0]
    #     n_rating_avg = n_rating_stats.n_rating_avg
    #     n_rating_min = n_rating_stats.n_rating_min
    #     n_rating_max = n_rating_stats.n_rating_max
    #     percentiles = self.popular_movies_df.approxQuantile("n_rating", [0.25, 0.50, 0.75], 0)
    #     n_rating_25_percentile, n_rating_50_percentile, n_rating_75_percentile = percentiles
    #     return {
    #         "n_rating_avg": n_rating_avg,
    #         "n_rating_min": n_rating_min,
    #         "n_rating_max": n_rating_max,
    #         "n_rating_25_percentile": n_rating_25_percentile,
    #         "n_rating_50_percentile": n_rating_50_percentile,
    #         "n_rating_75_percentile": n_rating_75_percentile,
    #     }

    # def rank_movies_by_number_of_rating(self, min_n_rating_threshold: Optional[int] = 0):
    #     popular_movies_df = self.preprocess_movies_df(min_n_rating_threshold)
    #     popular_movies_by_number_of_ratings_df = popular_movies_df.sort(F.desc("n_rating"))
    #     return popular_movies_by_number_of_ratings_df

    # def rank_movies_by_average_rating(self, min_n_rating_threshold: Optional[int] = None):
    #     popular_movies_df = self.preprocess_movies_df(min_n_rating_threshold)
    #     popular_movies_by_average_ratings_df = popular_movies_df.sort(F.desc("avg_rating"))
    #     return popular_movies_by_average_ratings_df

    # def rank_movies_by_std_rating(self, min_n_rating_threshold: Optional[int] = None):
    #     popular_movies_df = self.preprocess_movies_df(min_n_rating_threshold)
    #     popular_movies_by_std_ratings_df = popular_movies_df.filter(self.popular_movies_df.std_rating != np.nan).sort(F.desc("std_rating"))
    #     return popular_movies_by_std_ratings_df

    # def preprocess_movies_df(self, min_n_rating_threshold: Optional[int] = 0):
    #     if min_n_rating_threshold == 0:
    #         return self.popular_movies_df
    #     else:
    #         return self.popular_movies_df.filter(self.popular_movies_df.n_rating >= min_n_rating_threshold)


def compute_movielens_statistics(movielens20m: MovieLens20m):
    movielens_statistics = MovieLensStatistics(movielens20m)
    percenties = movielens_statistics.visualise_movie_n_rating_boxplot()
    print("Percenties: ", percenties)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    compute_movielens_statistics(movielens20m)
