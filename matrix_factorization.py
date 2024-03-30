from typing import Optional

import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

from constant import RANDOM_SEED
from dataset import MovieLens20m


class MatrixFactorization:
    def __init__(
        self,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating",
        rank: int = 5,
        max_iter: int = 10,
        seed: int = RANDOM_SEED,
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


def recommend_movies_by_matrix_factorization(movielens20m: MovieLens20m, threshold=3.5):
    ratings_df = movielens20m.get_ratings_df()

    mf = MatrixFactorization()
    ratings_df = mf.preprocess(ratings_df, min_n_rating_threshold=200)

    train, test = ratings_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)
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
    spark = SparkSession.builder.appName("CS5344 Project Matrix Factorization").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    recommend_movies_by_matrix_factorization(movielens20m, threshold=3.5)
