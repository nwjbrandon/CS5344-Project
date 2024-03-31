import os

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
