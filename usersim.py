from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, udf, explode
from math import sqrt
from pyspark.sql.types import FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.linalg import VectorUDT


spark=SparkSession.builder.appName('UserNB').getOrCreate()
spark.conf.set("spark.sql.pivotMaxValues", "27000")
spark.sparkContext.setLogLevel("WARN")


ratings_df = spark.read.option('header', 'true').csv('ml-20m/ratings.csv', inferSchema=True)
ratings_df = ratings_df.select('userId', 'movieId', 'rating')

# distinct_movie_count =  ratings_df.select('userId').distinct().count()
# print(f"The number of distinct users: {distinct_movie_count}")
# DISTINCT MOVIES = 27K, USERS = 138K


pivot_df = ratings_df.groupby('userId').pivot('movieId').avg('rating')
pivot_df = pivot_df.fillna(0)

# for printing a 10x10 snippet of pivotdf for debug
def printsnip(datfr):
    allcols = datfr.columns
    firsttencols = allcols[:10]
    datfr.limit(10).select(firsttencols).show()

def printlast(datfr):
    allcols = datfr.columns
    firsttencols = allcols[-1]
    datfr.limit(10).select(firsttencols).show()

# printsnip(pivot_df)


vector_col = "features"
assembler = VectorAssembler(inputCols=pivot_df.columns[1:], outputCol=vector_col)
# vector_df = assembler.transform(pivot_df).select('userId', vector_col)
vector_df = assembler.transform(pivot_df)


# printsnip(vector_df)
# printlast(vector_df)
# print("(*****************)")
# print(vector_df.select(vector_col).take(1))

# Normalize the feature vectors
def normalize_sparse_vector(v):
    norm = sqrt(sum([x**2 for x in v.values]))
    return SparseVector(v.size, v.indices, [x / norm for x in v.values])

normalize_sparse_vector_udf = udf(normalize_sparse_vector, VectorUDT())
normalized_df = vector_df.withColumn("norm_features", normalize_sparse_vector_udf("features"))

# print("(*****************)")
# printlast(normalized_df)


# Fit the ALS model
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop", implicitPrefs=False)
model = als.fit(ratings_df)

# Generate recommendations for all users
user_recs = model.recommendForAllUsers(10)

# Filter for the specific user and explode the recommendations
user_id_for_recommendations = 1  # The user ID we want recommendations for

recommendations_for_user = user_recs.filter(col("userId") == user_id_for_recommendations) \
                                    .withColumn("recommendations", explode("recommendations")) \
                                    .select(col("userId"), col("recommendations.*"))

# Show the recommendations for the user
recommendations_for_user.show()

# Stop the Spark session
spark.stop()