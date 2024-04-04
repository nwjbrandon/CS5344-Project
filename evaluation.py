from pyspark.sql.window import Window, WindowSpec
from pyspark.mllib.evaluation import RankingMetrics

import pyspark.sql.functions as F

def calculate_hit_rate(predictions, recommendation_count, threshold_rating):
    # Get top N recommendations for each user
    windowSpec = Window.partitionBy('userId').orderBy(F.desc('prediction'))
    top_prediction = predictions.withColumn('rank', F.rank().over(windowSpec)).filter(F.col('rank') <= recommendation_count)

    # Defining a hit as predicted rating above a certain threshold
    hits_per_user = top_prediction.withColumn('hit', (F.col('rating')>=threshold_rating).cast('int')).groupBy('userId').agg(F.sum('hit').alias('hits'))

    hits_per_user.show(10)
    # A user is considered a hit if they have at least 1 hit in their top N
    users_hit = hits_per_user.filter(F.col('hits')>0).count()

    # Total number of users who received recommendations
    total_users = predictions.select('userId').distinct().count()
    
    # Calculating hit ratio
    hit_rate = users_hit/total_users if total_users > 0 else 0
    # print(f"Hit Rate inside function: {hit_rate}")
    return hit_rate

def RMetrics(predictions, recommendation_count, threshold_rating):
    windowSpec = Window.partitionBy("userId").orderBy(F.desc("prediction"))
    # list of predicted top-N rated items per user
    predictedTopN = predictions.withColumn("rank", F.rank().over(windowSpec)).where(F.col("rank") <= recommendation_count).groupBy("userId").agg(F.collect_list("movieId").alias("predictedTopN"))

    # list of actual top-N rated items per user
    actualTopN = predictions.where(F.col("rating")>=threshold_rating).withColumn("rank", F.rank().over(windowSpec)).where(F.col("rank")<=recommendation_count).groupBy("userId").agg(F.collect_list("movieId").alias("actualTopN")) 

    topData = predictedTopN.join(actualTopN, "userId")

    topData_rdd = topData.rdd.map(lambda row: (row.predictedTopN, row.actualTopN))
    metrics = RankingMetrics(predictionAndLabels=topData_rdd)
    map = metrics.meanAveragePrecision
    precision = metrics.precisionAt(recommendation_count)
    ndcg = metrics.ndcgAt(recommendation_count)
    # recall = metrics.recallAt(recommendation_count)

    return map, precision, ndcg
    