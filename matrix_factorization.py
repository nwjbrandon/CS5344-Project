import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

from constants import SEED
from dataset import MovieLens20m
from evaluation import SparkRankingEvaluation, SparkRatingEvaluation


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
            rank=10,
            maxIter=15,
            implicitPrefs=False,
            regParam=0.05,
            coldStartStrategy="drop",
            nonnegative=False,
            seed=42,
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
    train, test = ratings_df.randomSplit([0.75, 0.25], seed=SEED)

    mf.fit(train)
    predictions = mf.predict(test)
    # predictions = predictions.filter(predictions.prediction != np.nan)

    rating_true = predictions.select(["userId", "movieId", "rating"])
    rating_pred = predictions.select(["userId", "movieId", "prediction"])

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

    # Ranking k = 10
    ranking_evaluator = SparkRankingEvaluation(
        rating_true,
        rating_pred,
        col_user="userId",
        col_item="movieId",
        col_rating="rating",
        col_prediction="prediction",
        k=10,
    )
    ranking_scores_at_10 = {
        "precision_at_10": ranking_evaluator.precision_at_k(),
        "recall_at_10": ranking_evaluator.recall_at_k(),
        "ndcg_at_10": ranking_evaluator.ndcg_at_k(),
        "map_at_10": ranking_evaluator.map_at_k(),
        "map": ranking_evaluator.map(),
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

    # rating_scores: {'rmse': 0.7921513406842974, 'mae': 0.6115370079657217, 'rsquared': 0.4328377265483636, 'exp_var': 0.44136285721742907}
    # ranking_scores_at_10: {'precision_at_10': 0.8937655283989139, 'recall_at_10': 0.5938712210786969, 'ndcg_at_10': 1.0, 'map_at_10': 1.0, 'map': 0.5938712210786972}
    # ranking_scores_at_20: {'precision_at_20': 0.7236158057433407, 'recall_at_20': 0.7695308541544534, 'ndcg_at_20': 1.0, 'map_at_20': 1.0, 'map': 0.7695308541544527}
    # ranking_scores_at_50: {'precision_at_50': 0.46414398798174156, 'recall_at_50': 0.9182487239618025, 'ndcg_at_50': 1.0, 'map_at_50': 1.0, 'map': 0.9182487239618027}
    # ranking_scores_at_100: {'precision_at_100': 0.2935147628127344, 'recall_at_100': 0.9725313614950274, 'ndcg_at_100': 1.0, 'map_at_100': 1.0, 'map': 0.9725313614950267}
    print("rating_scores:", rating_scores)
    print("ranking_scores_at_10:", ranking_scores_at_10)
    print("ranking_scores_at_20:", ranking_scores_at_20)
    print("ranking_scores_at_50:", ranking_scores_at_50)
    print("ranking_scores_at_100:", ranking_scores_at_100)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project Matrix Factorization").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    recommend_movies_by_matrix_factorization(movielens20m)
