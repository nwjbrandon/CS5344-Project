from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("Movie Recommendation System").getOrCreate()


# Load ratings data
ratings = spark.read.csv("ml-20m/ratings.csv", header=True, inferSchema=True)

# Load movies data
movies = spark.read.csv("ml-20m/movies.csv", header=True, inferSchema=True)

# Remove duplicates and handle missing values if any
ratings = ratings.dropDuplicates().na.drop()
movies = movies.dropDuplicates().na.drop()

# Assuming you're going to use the 'userId', 'movieId', and 'rating' columns for recommendations
ratings = ratings.select("userId", "movieId", "rating")
movies = movies.select("movieId", "title")

# Split the data into training and test sets
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the ALS model
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)

# Fit the model to the training data
model = als.fit(training)

# Predictions
predictions = model.transform(test)

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)

# Show recommendations for a specific user, for example, userId = 123
userRecs.filter(userRecs.userId == 123).select("recommendations").show(truncate=False)
