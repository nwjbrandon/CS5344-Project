from pyspark.sql.window import Window, WindowSpec

def calculate_hit_rate(predictions, recommendation_count = 10):
    # Get top N recommendations for each user
    windowSpec = Window.partitionBy('userId').orderBy(F.desc('prediction'))
    top_prediction = predictions.withColumn('rank', F.rank().over(windowSpec)).filter(F.col('rank') <= recommendation_count)

    # Defining a hit as predicted rating above a certain threshold
    hit_threshold = 3.5
    hits_per_user = top_prediction.withColumn('hit', (F.col('rating')>=hit_threshold).cast('int')).groupBy('userId').agg(F.sum('hit').alias('hits'))

    hits_per_user.show(10)
    # A user is considered a hit if they have at least 1 hit in their top N
    users_hit = hits_per_user.filter(F.col('hits')>0).count()

    # Total number of users who received recommendations
    total_users = predictions.select('userId').distinct().count()
    
    # Calculating hit ratio
    hit_rate = users_hit/total_users if total_users > 0 else 0
    # print(f"Hit Rate inside function: {hit_rate}")
    return hit_rate