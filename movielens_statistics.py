import matplotlib.pyplot as plt
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from dataset import MovieLens20m


class MovieLensStatistics:
    def __init__(self, movielens20m: MovieLens20m) -> None:
        self.movielens20m = movielens20m
        self.ratings_df = self.movielens20m.get_ratings_df()
        self.movies_df = self.movielens20m.get_movies_df()
        self.features = self.movielens20m.extract_features()

    def get_dataset_sizes(self):
        number_of_ratings = self.ratings_df.count()
        number_of_movies = self.movies_df.count()

        return {
            "number_of_ratings": number_of_ratings,
            "number_of_movies": number_of_movies,
        }

    def get_timespan_of_ratings(self):
        ratings_df = self.ratings_df
        timestamps = ratings_df.select([F.min("timestamp").alias("min_timestamp"), F.max("timestamp").alias("max_timestamp")]).collect()

        min_timestamp = timestamps[0].min_timestamp
        max_timestamp = timestamps[0].max_timestamp

        timespan_in_days = (max_timestamp - min_timestamp) / 60 / 60 / 24
        timespan_in_years = timespan_in_days / 365.25

        return {
            "timespan_in_days": timespan_in_days,
            "timespan_in_years": timespan_in_years,
        }

    def visualise_boxplot_of_number_of_rating_per_movie(self):
        df = self.movies_df.join(self.features["ratings_per_movie_df"], ["movieID"])
        percentiles = df.approxQuantile("number_of_rating_per_movie", [0.00, 0.25, 0.50, 0.75, 1.00], 0)
        whislo = percentiles[0]
        q1 = percentiles[1]
        med = percentiles[2]
        q3 = percentiles[3]
        whishi = percentiles[4]
        mean = df.select([F.mean("number_of_rating_per_movie").alias("mean")]).collect()[0]["mean"]

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

    def visualise_boxplot_of_number_of_rating_per_user(self):
        df = self.features["ratings_per_user_df"]
        percentiles = df.approxQuantile("number_of_rating_per_user", [0.00, 0.25, 0.50, 0.75, 1.00], 0)
        whislo = percentiles[0]
        q1 = percentiles[1]
        med = percentiles[2]
        q3 = percentiles[3]
        whishi = percentiles[4]
        mean = df.select([F.mean("number_of_rating_per_user").alias("mean")]).collect()[0]["mean"]

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

    def visualise_barplot_of_genres(self):
        movies_df = self.movies_df
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

    def visualise_barplot_of_genres_rating(self):
        ratings_per_movie_df = self.features["ratings_per_movie_df"]
        df = ratings_per_movie_df.join(self.movies_df, ["movieId"])

        df = df.select(["genres", "average_rating_per_movie"])
        df = df.select(F.explode(F.split(F.col("genres"), "\\|")).alias("genres"), "average_rating_per_movie")
        df = df.groupBy("genres").agg(F.avg("average_rating_per_movie").alias("rating"))

        df = df.toPandas()
        df = df.set_index("genres")
        df.plot.bar()

        plt.title("Bar Plot Of Ratings Of Genres Distribution In MovieLens20m")
        plt.ylabel("Average Rating")
        plt.savefig("imgs/genres_rating_distribution.png", bbox_inches="tight")
        plt.close()

    def visualise_scatterplot_of_number_of_rating_per_movie_over_movie_age(self, current_year=2024):
        movies_df = self.movies_df
        ratings_per_movie_df = self.features["ratings_per_movie_df"]
        movie_age_df = self.features["movie_age_df"]

        df = movies_df.join(ratings_per_movie_df, ["movieID"])
        df = df.join(movie_age_df, ["movieID"])

        df = df.filter(df.movie_age != np.nan)
        df = df.filter(df.movie_age < 200)

        df = df.select(["movieId", "title", "number_of_rating_per_movie", "movie_age"])

        df = df.toPandas()
        plt.scatter(x=df["movie_age"], y=df["number_of_rating_per_movie"], s=2)
        plt.title("Scatter Plot Of Number Of Rating And Movie Age")
        plt.xlabel("Movie Age (years)")
        plt.ylabel("Number Of Ratings")
        plt.savefig("imgs/scatterplot_of_n_rating_and_movie_age.png", bbox_inches="tight")
        plt.close()

    def visualise_genre_trends(self):
        ratings_df = self.ratings_df
        movies_df = self.movies_df
        movie_age_df = self.features["movie_age_df"]
        df = ratings_df.join(movies_df, ["movieId"])
        df = df.join(movie_age_df, ["movieId"])

        df = df.filter(df.movie_age != np.nan)
        df = df.filter(df.movie_age < 200)

        df = df.filter(df.genres != "(no genres listed)")
        df = df.select(["movieId", "userId", "released_year", F.explode(F.split(F.col("genres"), "\\|")).alias("genres")])

        df = df.groupBy(["released_year", "genres"]).count()
        df = df.sort(F.asc("genres"), F.asc("released_year"))

        df = df.toPandas()
        genres = np.unique(df["genres"]).tolist()
        plt.figure(figsize=(16, 8))
        for genre in genres:
            sub_df = df[df["genres"] == genre]
            plt.plot(sub_df["released_year"], sub_df["count"], label=genre)
        plt.legend()
        plt.title("Trends In Genres Over The Years Measured By Number Of Ratings")
        plt.ylabel("Number Of Ratings")
        plt.xlabel("Years")
        plt.savefig("imgs/genre_trends.png")
        plt.close()

    def visualise_genres_barplot_for_user(self, user_id=5):
        movies_df = self.movies_df
        ratings_df = self.ratings_df
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


def compute_movielens_statistics(movielens20m: MovieLens20m):
    movielens_statistics = MovieLensStatistics(movielens20m)

    dataset_sizes = movielens_statistics.get_dataset_sizes()
    print("dataset_sizes: ", dataset_sizes)

    timespan = movielens_statistics.get_timespan_of_ratings()
    print("timespan:", timespan)

    movie_rating_statistics = movielens_statistics.visualise_boxplot_of_number_of_rating_per_movie()
    print("movie_rating_statistics: ", movie_rating_statistics)

    user_rating_statistics = movielens_statistics.visualise_boxplot_of_number_of_rating_per_user()
    print("user_rating_statistics: ", user_rating_statistics)

    genres_statistics = movielens_statistics.visualise_barplot_of_genres()
    print("genres_statistics: ", genres_statistics)

    movielens_statistics.visualise_barplot_of_genres_rating()

    movielens_statistics.visualise_scatterplot_of_number_of_rating_per_movie_over_movie_age()

    movielens_statistics.visualise_genre_trends()

    movielens_statistics.visualise_genres_barplot_for_user(user_id=3)
    movielens_statistics.visualise_genres_barplot_for_user(user_id=5)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project MovieLens Statistics").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    compute_movielens_statistics(movielens20m)
