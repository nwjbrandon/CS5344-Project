import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

from constants import DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL, SEED
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
            maxIter=20,
            implicitPrefs=False,
            alpha=0.1,
            regParam=0.05,
            coldStartStrategy="drop",
            nonnegative=False,
            seed=42,
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
        )
        self.model = None

    def fit(self, train):
        self.model = self.als.fit(train.select([self.user_col, self.item_col, self.rating_col]))

    def predict(self, test):
        predictions = self.model.transform(test.select([self.user_col, self.item_col]))
        predictions = predictions.join(test, [self.user_col, self.item_col])
        return predictions

    def get_top_k_recommendations(self, train):
        """
        Reference: https://github.com/recommenders-team/recommenders
        """
        # Get the cross join of all user-item pairs and score them.
        users = train.select(DEFAULT_USER_COL).distinct()
        items = train.select(DEFAULT_ITEM_COL).distinct()
        user_item = users.crossJoin(items)
        dfs_pred = self.model.transform(user_item)

        # Remove seen items
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(train.alias("train"), (dfs_pred[DEFAULT_USER_COL] == train[DEFAULT_USER_COL]) & (dfs_pred[DEFAULT_ITEM_COL] == train[DEFAULT_ITEM_COL]), how="outer")
        top_k_recommendations = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train." + DEFAULT_RATING_COL].isNull()).select("pred." + DEFAULT_USER_COL, "pred." + DEFAULT_ITEM_COL, "pred." + DEFAULT_PREDICTION_COL)
        return top_k_recommendations

    def evaluate(self, train, test):
        rating_scores = self.evaluate_rating(test)
        ranking_scores_10 = self.evaluate_ranking(train, test, k=10)
        # ranking_scores_20 = self.evaluate_ranking(train, test, k=20)
        # ranking_scores_50 = self.evaluate_ranking(train, test, k=50)
        # ranking_scores_100 = self.evaluate_ranking(train, test, k=100)
        return {
            **rating_scores,
            **ranking_scores_10,
            # **ranking_scores_20,
            # **ranking_scores_50,
            # **ranking_scores_100,
        }

    def evaluate_rating(self, test):
        predictions = self.predict(test)
        rating_true = predictions.select(["userId", "movieId", "rating"])
        rating_pred = predictions.select(["userId", "movieId", "prediction"])

        rating_evaluator = SparkRatingEvaluation(
            rating_true,
            rating_pred,
            col_user="userId",
            col_item="movieId",
            col_rating="rating",
            col_prediction="prediction",
        )
        return {
            "rmse": rating_evaluator.rmse(),
            "mae": rating_evaluator.mae(),
            "rsquared": rating_evaluator.rsquared(),
            "exp_var": rating_evaluator.exp_var(),
        }

    def evaluate_ranking(self, train, test, k):
        top_k_recommendations = self.get_top_k_recommendations(train)
        ranking_evaluator = SparkRankingEvaluation(
            test,
            top_k_recommendations,
            col_user="userId",
            col_item="movieId",
            col_rating="rating",
            col_prediction="prediction",
            k=k,
        )
        return {
            f"precision_at_{k}": ranking_evaluator.precision_at_k(),
            f"recall_at_{k}": ranking_evaluator.recall_at_k(),
            f"ndcg_at_{k}": ranking_evaluator.ndcg_at_k(),
            f"map_at_{k}": ranking_evaluator.map_at_k(),
            "map": ranking_evaluator.map(),
        }


def recommend_movies_by_matrix_factorization(movielens20m: MovieLens20m):
    dfs = movielens20m.extract_features()

    ratings_df = movielens20m.get_ratings_df()
    ratings_per_movie_df = dfs["ratings_per_movie_df"]
    ratings_per_user_df = dfs["ratings_per_user_df"]
    movie_age_df = dfs["movie_age_df"]

    df = ratings_df.join(ratings_per_movie_df, ["movieID"])
    df = df.join(ratings_per_user_df, ["userID"])
    df = df.join(movie_age_df, ["movieID"])

    df = df.filter(df["number_of_rating_per_movie"] >= 20)
    df = df.filter(df["number_of_rating_per_user"] >= 30)
    df = df.filter(df["movie_age"] != np.nan)
    df = df.filter(df["movie_age"] <= 100)
    df.show()

    mf = MatrixFactorization()
    train, test = df.randomSplit([0.75, 0.25], seed=SEED)
    mf.fit(train)
    scores = mf.evaluate(train, test)
    print(scores)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project Matrix Factorization").getOrCreate()
    movielens20m = MovieLens20m(spark=spark)
    recommend_movies_by_matrix_factorization(movielens20m)
