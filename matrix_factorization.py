import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, IntegerType

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

    def fit(self, df):
        self.model = self.als.fit(df.select([self.user_col, self.item_col, self.rating_col]))

    def predict(self, df):
        predictions = self.model.transform(df.select([self.user_col, self.item_col]))
        predictions = predictions.join(df, [self.user_col, self.item_col])
        return predictions

    def evaluate(self, df, movielens20m: MovieLens20m, recommendation_count):
        evaluator = Evaluator(df, movielens20m)
        return evaluator.evaluate(recommendation_count)

    def recommend_for_all_users(self, k):
        return self.model.recommendForAllUsers(k)


@F.udf(returnType=ArrayType(IntegerType()))
def get_recommended_movie_ids(recommendations):
    return [recommendation["movieId"] for recommendation in recommendations]


def recommend_movies_by_matrix_factorization(movielens20m: MovieLens20m):
    dfs = movielens20m.extract_features()

    ratings_df = movielens20m.get_ratings_df()
    ratings_per_movie_df = dfs["ratings_per_movie_df"]
    ratings_per_user_df = dfs["ratings_per_user_df"]
    movie_age_df = dfs["movie_age_df"]

    df = ratings_df.join(ratings_per_movie_df, ["movieID"])
    df = df.join(ratings_per_user_df, ["userID"])
    df = df.join(movie_age_df, ["movieID"])

    df = df.filter(df["number_of_rating_per_movie"] >= 10)
    df = df.filter(df["number_of_rating_per_user"] >= 30)
    df = df.filter(df["movie_age"] != np.nan)
    df = df.filter(df["movie_age"] <= 100)

    df.show()  # 19332208

    mf = MatrixFactorization()
    train, test = ratings_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)

    mf.fit(train)
    predictions = mf.predict(test)
    valid_predictions = predictions.filter(predictions.prediction != np.nan)
    scores = mf.evaluate(valid_predictions, movielens20m, recommendation_count=10)
    # {'rmse': 0.8174848413520778, 'hit_rate': 0.5820735777059817, 'coverage': 0.63314759146565, 'Mean Average Precision': 0.8521349892412887, 'Precision': 0.5096894848270992, 'NDCG': 0.9081199219876757}
    print(scores)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project Matrix Factorization").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    recommend_movies_by_matrix_factorization(movielens20m)
