from typing import Optional

import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

from constant import RANDOM_SEED
from dataset import MovieLens20m
from evaluator import Evaluator


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

    def evaluate(self, df, movielens20m: MovieLens20m, recommendation_count):
        evaluator = Evaluator(df, movielens20m)
        return evaluator.evaluate(recommendation_count)


def recommend_movies_by_matrix_factorization(movielens20m: MovieLens20m):
    ratings_df = movielens20m.get_ratings_df()

    n_rating_50_percentile = 18
    mf = MatrixFactorization()
    ratings_df = mf.preprocess(ratings_df, min_n_rating_threshold=n_rating_50_percentile)

    train, test = ratings_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)
    mf.fit(train)
    predictions = mf.predict(test)

    # Filter rows where predictions are not NaN
    valid_predictions = predictions.filter(predictions.prediction != np.nan)
    valid_predictions.show(10)
    recommendation_count = 10
    scores = mf.evaluate(valid_predictions, movielens20m, recommendation_count)
    # {'rmse': 0.8165847881901006, 'hit_rate': 0.40415162228417006, 'coverage': 0.485996040765452, 
    # 'Mean Average Precision': 0.8516845095504275, 'Precision': 0.5089423189881344, 'NDCG': 0.9078648884072624}
    print(scores)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project Matrix Factorization").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    recommend_movies_by_matrix_factorization(movielens20m)
