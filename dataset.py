import os

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType

schema_ratings = StructType(
    [
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("rating", FloatType(), False),
        StructField("timestamp", IntegerType(), False),
    ]
)

schema_movies = StructType(
    [
        StructField("movieId", IntegerType(), False),
        StructField("title", StringType(), False),
        StructField("genres", StringType(), False),
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

    def get_movies_df(self):
        return self.movies_df

    def get_ratings_df(self):
        return self.ratings_df

    def extract_features(self):
        return {
            "movie_age_df": self.get_movie_age(),
            "ratings_per_movie_df": self.get_ratings_per_movie(),
            "ratings_per_user_df": self.get_ratings_per_user(),
        }

    def get_ratings_per_movie(self):
        return self.ratings_df.groupBy("movieId").agg(F.count("userId").alias("number_of_rating_per_movie"), F.avg("rating").alias("average_rating_per_movie"), F.stddev("rating").alias("standard_deviation_rating_per_movie"))

    def get_ratings_per_user(self):
        return self.ratings_df.groupBy("userId").agg(F.count("userId").alias("number_of_rating_per_user"))

    def get_movie_age(self, current_year=2024):
        df = self.movies_df.select(["movieID", "title"])
        df = df.withColumn("released_year", F.regexp_extract(F.col("title"), r"(?<=\()(\d+)(?=\))", 1).cast(IntegerType()))
        df = df.withColumn("movie_age", current_year - df["released_year"])
        df = df.select(["movieID", "released_year", "movie_age"])
        return df
