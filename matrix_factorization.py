import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

from constant import RANDOM_SEED
from dataset import MovieLens20m
from evaluation import SparkDiversityEvaluation, SparkRankingEvaluation, SparkRatingEvaluation


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

    # 19332208
    df.show()

    mf = MatrixFactorization()
    train, test = ratings_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)

    mf.fit(train)
    predictions = mf.predict(test)
    valid_predictions = predictions.filter(predictions.prediction != np.nan)

    rating_true = valid_predictions.select(["userId", "movieId", "rating"])
    rating_pred = valid_predictions.select(["userId", "movieId", "prediction"])

    # Rating
    rating_evaluator = SparkRatingEvaluation(
        rating_true,
        rating_pred,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
    )
    rating_scores = {
        "rmse": rating_evaluator.rmse(),
        "mae": rating_evaluator.mae(),
        "rsquared": rating_evaluator.rsquared(),
        "exp_var": rating_evaluator.exp_var(),
    }

    # Ranking k = 20
    ranking_evaluator = SparkRankingEvaluation(
        rating_true,
        rating_pred,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
        k=20,
    )
    ranking_scores_at_20 = {
        "precision_at_20": ranking_evaluator.precision_at_k(),
        "recall_at_20": ranking_evaluator.recall_at_k(),
        "ndcg_at_20": ranking_evaluator.ndcg_at_k(),
        "map_at_20": ranking_evaluator.map_at_k(),
        "map": ranking_evaluator.map(),
    }

    # Ranking k = 50
    ranking_evaluator = SparkRankingEvaluation(
        rating_true,
        rating_pred,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
        k=50,
    )
    ranking_scores_at_50 = {
        "precision_at_50": ranking_evaluator.precision_at_k(),
        "recall_at_50": ranking_evaluator.recall_at_k(),
        "ndcg_at_50": ranking_evaluator.ndcg_at_k(),
        "map_at_50": ranking_evaluator.map_at_k(),
        "map": ranking_evaluator.map(),
    }

    # Ranking k = 100
    ranking_evaluator = SparkRankingEvaluation(
        rating_true,
        rating_pred,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
        k=100,
    )
    ranking_scores_at_100 = {
        "precision_at_100": ranking_evaluator.precision_at_k(),
        "recall_at_100": ranking_evaluator.recall_at_k(),
        "ndcg_at_100": ranking_evaluator.ndcg_at_k(),
        "map_at_100": ranking_evaluator.map_at_k(),
        "map": ranking_evaluator.map(),
    }

    # rating_scores: {'rmse': 0.8125284521121838, 'mae': 0.6297998826940571, 'rsquared': 0.4035379062385661, 'exp_var': 0.4112955720459812}
    # ranking_scores_at_20: {'precision_at_20': 0.47074078715546286, 'recall_at_20': 0.9155656907613267, 'ndcg_at_20': 1.0, 'map_at_20': 1.0, 'map': 0.9155656907613265}
    # ranking_scores_at_50: {'precision_at_50': 0.25243643708579966, 'recall_at_50': 0.9814030788264771, 'ndcg_at_50': 1.0, 'map_at_50': 1.0, 'map': 0.981403078826477}
    # ranking_scores_at_100: {'precision_at_100': 0.14064465953572716, 'recall_at_100': 0.9963873027081633, 'ndcg_at_100': 1.0, 'map_at_100': 1.0, 'map': 0.9963873027081638}
    print("rating_scores:", rating_scores)
    print("ranking_scores_at_20:", ranking_scores_at_20)
    print("ranking_scores_at_50:", ranking_scores_at_50)
    print("ranking_scores_at_100:", ranking_scores_at_100)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project Matrix Factorization").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    recommend_movies_by_matrix_factorization(movielens20m)
