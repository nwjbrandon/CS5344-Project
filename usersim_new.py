from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from pyspark.ml.recommendation import ALS

spark=SparkSession.builder.appName('UserNB').getOrCreate()
spark.conf.set("spark.sql.pivotMaxValues", "27000")
spark.sparkContext.setLogLevel("WARN")


ratings_df = spark.read.option('header', 'true').csv('ml-20m/ratings.csv', inferSchema=True)
ratings_df = ratings_df.select('userId', 'movieId', 'rating')

ratings_df = ratings_df.limit(10000)


# distinct_movie_count =  ratings_df.select('userId').distinct().count()
# print(f"The number of distinct users: {distinct_movie_count}")
# DISTINCT MOVIES = 27K, USERS = 138K


pivot_df = ratings_df.groupby('userId').pivot('movieId').avg('rating')
pivot_df = pivot_df.fillna(0)

#for printing a 10x10 snippet of pivotdf for debug
# def printsnip(datfr):
#     allcols = datfr.columns
#     firsttencols = allcols[:10]
#     datfr.limit(10).select(firsttencols).show()

# def printlast(datfr):
#     allcols = datfr.columns
#     firsttencols = allcols[-1]
#     datfr.limit(10).select(firsttencols).show()

# printsnip(pivot_df)


vector_col = "vec_features"
assembler = VectorAssembler(inputCols=pivot_df.columns[1:], outputCol=vector_col)
vector_df = assembler.transform(pivot_df).select('userId', vector_col)
# vector_df = assembler.transform(pivot_df)


# printsnip(vector_df)
# printlast(vector_df)

print("(*****************)")
print(vector_df.select(vector_col)[0])

