import os
from typing import Optional

import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType

SEED = 42

schema_ratings = StructType(
    [
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("rating", FloatType(), True),
        StructField("timestamp", StringType(), True),
    ]
)

schema_movies = StructType(
    [
        StructField("movieId", IntegerType(), False),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True),
    ]
)


class MovieLens20m:
    def __init__(self, spark, data_dir: str = "ml-20m") -> None:
        self.spark = spark
        self.movie_fname = os.path.join(data_dir, "movies.csv")
        self.ratings_fname = os.path.join(data_dir, "ratings.csv")

        # Load datasets
        self.movies_df = self.spark.read.option("header", True).schema(schema_movies).csv(self.movie_fname)
        self.ratings_df = self.spark.read.option("header", True).schema(schema_ratings).csv(self.ratings_fname)

        # Compute movie popularity and rating statistics
        self.popular_movies_df = self.compute_movie_popularity()
        self.rating_statistics = self.compute_rating_statistics()

    def get_movies_df(self):
        return self.movies_df

    def get_ratings_df(self):
        return self.ratings_df

    def get_popular_movies_df(self):
        return self.popular_movies_df

    def compute_movie_popularity(self):
        popular_movies_df = self.ratings_df.groupBy("movieId").agg(F.count("userId").alias("n_rating"), F.avg("rating").alias("avg_rating"), F.stddev("rating").alias("std_rating"))
        popular_movies_df = popular_movies_df.join(self.movies_df, ["movieId"])
        return popular_movies_df

    def compute_rating_statistics(self):
        n_rating_stats = self.popular_movies_df.select(
            [
                F.mean("n_rating").alias("n_rating_avg"),
                F.min("n_rating").alias("n_rating_min"),
                F.max("n_rating").alias("n_rating_max"),
            ]
        ).collect()
        if len(n_rating_stats) == 0:
            raise "Not Enough Data"

        n_rating_stats = n_rating_stats[0]
        n_rating_avg = n_rating_stats.n_rating_avg
        n_rating_min = n_rating_stats.n_rating_min
        n_rating_max = n_rating_stats.n_rating_max
        percentiles = self.popular_movies_df.approxQuantile("n_rating", [0.25, 0.50, 0.75], 0)
        n_rating_25_percentile, n_rating_50_percentile, n_rating_75_percentile = percentiles
        return {
            "n_rating_avg": n_rating_avg,
            "n_rating_min": n_rating_min,
            "n_rating_max": n_rating_max,
            "n_rating_25_percentile": n_rating_25_percentile,
            "n_rating_50_percentile": n_rating_50_percentile,
            "n_rating_75_percentile": n_rating_75_percentile,
        }

    def get_rating_statistics(self):
        return self.rating_statistics

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


class MatrixFactorization:
    def __init__(
        self,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating",
        rank: int = 5,
        max_iter: int = 10,
        seed: int = SEED,
    ) -> None:
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.rank = rank
        self.max_iter = max_iter
        self.seed = seed
        self.prediction_col = "prediction"

        self.als = ALS(
            rank=self.rank,
            maxIter=self.max_iter,
            seed=self.seed,
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
        )
        self.model = None
        self.evaluator = RegressionEvaluator(predictionCol=self.prediction_col, labelCol=self.rating_col, metricName="rmse")

    def preprocess(self, df, min_n_rating_threshold: Optional[int] = 0):
        # Remove movie with few number of ratings
        n_rating_df = df.groupBy(self.item_col).agg(F.count(self.user_col).alias("n_rating"))
        df = df.join(n_rating_df, [self.item_col])
        df = df.filter(df.n_rating >= min_n_rating_threshold)
        return df

    def fit(self, df):
        self.model = self.als.fit(df.select([self.user_col, self.item_col, self.rating_col]))

    def predict(self, df):
        predictions = self.model.transform(df.select([self.user_col, self.item_col]))
        predictions = predictions.join(df, [self.user_col, self.item_col])
        return predictions

    def evaluate(self, df):
        return self.evaluator.evaluate(df)


def recommend_movies_by_popularity(movielens20m: MovieLens20m):
    # Compute rating statistics to understand the skew in the number of ratings
    rating_statistics = movielens20m.get_rating_statistics()

    print("rating_statistics:", rating_statistics)
    min_n_rating_threshold = rating_statistics["n_rating_75_percentile"]

    # Rank popular movies by the number of ratings
    movielens20m.rank_movies_by_number_of_rating(min_n_rating_threshold=min_n_rating_threshold).show(10)

    # Rank popular movies by the average ratings
    movielens20m.rank_movies_by_average_rating(min_n_rating_threshold=min_n_rating_threshold).show(10)

    # Rank popular movies that has polarized ratings
    movielens20m.rank_movies_by_std_rating(min_n_rating_threshold=min_n_rating_threshold).show(10)


def recommend_movies_by_matrix_factorization(movielens20m: MovieLens20m, threshold=3.5):
    ratings_df = movielens20m.get_ratings_df()
    rating_statistics = movielens20m.get_rating_statistics()
    min_n_rating_threshold = rating_statistics["n_rating_50_percentile"]

    mf = MatrixFactorization()
    ratings_df = mf.preprocess(ratings_df, min_n_rating_threshold=min_n_rating_threshold)

    train, test = ratings_df.randomSplit([0.9, 0.1], seed=SEED)
    mf.fit(train)
    predictions = mf.predict(test)

    # Filter rows where predictions are not NaN
    valid_predictions = predictions.filter(predictions.prediction != np.nan)

    # Calculate Hit Rate
    hits = valid_predictions.filter(valid_predictions.rating >= threshold).filter(valid_predictions.prediction >= threshold).count()
    total = valid_predictions.count()
    hit_rate = hits / total if total > 0 else 0

    # Calculate Coverage
    unique_recommended = valid_predictions.select("movieId").distinct().count()
    total_movies = movielens20m.get_movies_df().select("movieId").distinct().count()
    coverage = unique_recommended / total_movies if total_movies > 0 else 0

    valid_predictions.show(10)

    rmse = mf.evaluate(valid_predictions)
    print("RMSE:", rmse)  # 0.8165847881901006
    print(f"Hit Rate: {hit_rate}")
    print(f"Coverage: {coverage}")


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)

    recommend_movies_by_popularity(movielens20m)
    recommend_movies_by_matrix_factorization(movielens20m, threshold=3.5)
