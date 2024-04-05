from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from dataset import MovieLens20m
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from functools import lru_cache as lru

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
    
    def df_to_rdd(self, recommendation_count, threshold_rating = 3.5):
        windowSpec = Window.partitionBy("userId").orderBy(F.desc("prediction"))
        # list of predicted top-N rated items per user
        predictedTopN = self.prediction_df.withColumn("rank", F.rank().over(windowSpec)).where(F.col("rank") <= recommendation_count).groupBy("userId").agg(F.collect_list("movieId").alias("predictedTopN"))

        # list of actual top-N rated items per user
        actualTopN = self.prediction_df.where(F.col("rating")>=threshold_rating).withColumn("rank", F.rank().over(windowSpec)).where(F.col("rank")<=recommendation_count).groupBy("userId").agg(F.collect_list("movieId").alias("actualTopN")) 

        topData = predictedTopN.join(actualTopN, "userId")

        topData_rdd = topData.rdd.map(lambda row: (row.predictedTopN, row.actualTopN))

        return topData_rdd
    # def df_to_rdd(self, recommendation_count, threshold_rating = 3.5):
    #     windowSpec = Window.partitionBy("userId").orderBy(F.desc("prediction"))
    #     print(f"Recommendation count: {recommendation_count}, Type: {type(recommendation_count)}")
    #     print("predictiondfSchema")
    #     self.prediction_df.printSchema()
    #     print("predictiondf")
    #     self.prediction_df.show(5)
    #     debug_df = self.prediction_df.withColumn("rank", F.rank().over(windowSpec)).where(F.col("rank") <= recommendation_count)
    #     debug_df.show(10)
    #     return debug_df
    
    def meanAveragePrecision(self, topData):
        # topData = self.df_to_rdd(self.prediction_df)
        metrics = RankingMetrics(predictionAndLabels= topData)
        meanAveragePrecision = metrics.meanAveragePrecision
        return meanAveragePrecision
    
    def precision(self, topData, recommendation_count):
        # topData = self.df_to_rdd(self.prediction_df)
        metrics = RankingMetrics(predictionAndLabels= topData)
        precision = metrics.precisionAt(recommendation_count)

        return precision
    
    def ndcg(self, topData, recommendation_count):
        metrics = RankingMetrics(predictionAndLabels= topData)
        ndcg = metrics.ndcgAt(recommendation_count)

        return ndcg

    def evaluate(self, recommendation_count):
        topData_rdd = self.df_to_rdd(recommendation_count, threshold_rating = 3.5)
        return {
            "rmse": self.compute_rmse(),
            "hit_rate": self.compute_hit_rate(),
            "coverage": self.compute_coverage(),
            "Mean Average Precision": self.meanAveragePrecision(topData_rdd),
            "Precision": self.precision(topData_rdd, recommendation_count),
            "NDCG": self.ndcg(topData_rdd, recommendation_count)
        }
