from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

spark = SparkSession.builder.appName("MovieLens Clustering").getOrCreate()

# Load movies data
movies = spark.read.csv("ml-20m/movies.csv", header=True, inferSchema=True)

movies = movies.withColumn("genres", split(movies["genres"], "\\|"))

# Explode the genres array into a new row for each genre for each movie
# movies_exploded = movies.withColumn("genre", explode(movies["genres"]))

# Apply CountVectorizer to transform the genres into feature vectors
cv = CountVectorizer(inputCol="genres", outputCol="features", vocabSize=20, minDF=1.0)

# Fit and transform the CountVectorizer model to create the feature vectors
cv_model = cv.fit(movies)
movies_featured = cv_model.transform(movies)

# Group back the exploded rows into vectors per movie
# movies_grouped = movies_featured.groupBy("movieId", "title").agg(collect_list("features").alias("features"))

kmeans = KMeans().setK(20).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(movies_featured)

# Make predictions
predictions = model.transform(movies_featured)

predictions.select("title", "prediction").show()

### FOR MANUAL EVALUATION, UNCOMMENT CODE ###

# Display the most frequent genres in each cluster
# for i in range(20):  # Assuming we have 20 clusters
#     print(f"Cluster {i}:")
#     cluster_movies = predictions.filter(predictions.prediction == i)
#     genres = cluster_movies.withColumn("genres", explode("genres"))
#     genre_counts = genres.groupBy("genres").count().orderBy("count", ascending=False)
#     genre_counts.show()
