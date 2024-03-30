from pyspark.ml.evaluation import RegressionEvaluator

from dataset import MovieLens20m


class Evaluator:
    def __init__(
        self,
        prediction_df,
        movielens20m: MovieLens20m,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating",
        prediction_col: str = "prediction",
    ) -> None:
        self.prediction_df = prediction_df
        self.movielens20m = movielens20m

        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.prediction_col = prediction_col

    def compute_rmse(self):
        evaluator = RegressionEvaluator(predictionCol=self.prediction_col, labelCol=self.rating_col, metricName="rmse")
        return evaluator.evaluate(self.prediction_df)

    def compute_hit_rate(self, threshold: float = 3.5):
        hits = self.prediction_df.filter(self.prediction_df[self.rating_col] >= threshold).filter(self.prediction_df[self.prediction_col] >= threshold).count()
        total = self.prediction_df.count()
        hit_rate = hits / total if total > 0 else 0
        return hit_rate

    def compute_coverage(self):
        movies_df = self.movielens20m.get_movies_df()

        unique_recommended = self.prediction_df.select(self.item_col).distinct().count()
        total_movies = movies_df.select(self.item_col).distinct().count()
        coverage = unique_recommended / total_movies if total_movies > 0 else 0
        return coverage

    def evaluate(self):
        return {
            "rmse": self.compute_rmse(),
            "hit_rate": self.compute_hit_rate(),
            "coverage": self.compute_coverage(),
        }
