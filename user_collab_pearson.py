from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, udf, explode
from math import sqrt
from pyspark.sql.types import FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.linalg import VectorUDT
from pyspark.mllib.stat import Statistics
import numpy as np
import pandas as pd


from dataset import MovieLens20m


spark=SparkSession.builder.appName('UserNB').getOrCreate()
spark.conf.set("spark.sql.pivotMaxValues", "27000")
spark.sparkContext.setLogLevel("INFO")


# ratings_df = spark.read.option('header', 'true').csv('ml-20m/ratings.csv', inferSchema=True)
ratings_df = spark.read.option('header', 'true').csv('ml-1m/ratings.csv', inferSchema=True)
# ratings_df = spark.read.option('header', 'true').csv('ml-100k/ratings.csv', inferSchema=True)

movies_df = spark.read.option('header', 'true').csv('ml-100k/movies.csv', inferSchema=True)

ratings_df = ratings_df.select(col('userId'), col('movieId'), col('rating'))

# ratings_df = ratings_df.limit(20000)


distinct_movie_count =  ratings_df.select('movieId').distinct().count()
print(f"The number of distinct movies: {distinct_movie_count}")

distinct_user_count =  ratings_df.select('userId').distinct().count()
print(f"The number of distinct users: {distinct_user_count}")
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

# Convert to RDD for correlation calculation
rating_rdd = pivot_df.rdd.map(lambda row: (row[1:]))

# Assemble features into vector
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=pivot_df.columns[1:], outputCol=vector_col)
vector_matrix = assembler.transform(pivot_df)

# Compute Pearson correlation matrix
matrix = Correlation.corr(vector_matrix, vector_col, "pearson").head()

# printsnip(vector_matrix)
# print(matrix)

# Get the resulting correlation matrix as a Dense Matrix
correlation_matrix = matrix[0].toArray()

# print(correlation_matrix) #WORKS

def recommend_movies(user_id, top_n, rec_count, user_movie_matrix, correlation_matrix):
    user_index = user_movie_matrix.columns[1:].index(str(user_id))
    user_correlations = correlation_matrix[user_index]
    
    top_indices = np.argsort(user_correlations)[-top_n-1:-1][::-1]  # Exclude the user itself
    top_similarities = user_correlations[top_indices]
    
    # Fetch movies rated by similar users and recommend
    recommended_movies = {}
    movie_weights = {}
    movie_rating_count = {}

    for i, index in enumerate(top_indices):
        similar_user_id = int(user_movie_matrix.columns[1:][index])
        similarity_score = top_similarities[i]
        similar_user_ratings = ratings_df.filter(ratings_df["userId"] == similar_user_id)
        similar_user_movies = similar_user_ratings.select("movieId", "rating").collect()
        
        for movie in similar_user_movies:
            movie_id = movie.movieId
            weighted_rating = movie.rating * similarity_score

            if movie_id not in user_movie_matrix.columns:
                if movie_id not in recommended_movies:
                    recommended_movies[movie_id] = weighted_rating
                    movie_weights[movie_id] = similarity_score
                    movie_rating_count[movie_id] = 1
                    # recommended_movies[movie_id] = 1
                else:
                    recommended_movies[movie_id] += weighted_rating
                    movie_weights[movie_id] += similarity_score
                    movie_rating_count[movie_id] += 1
                    # recommended_movies[movie_id] += 1

        final_recommendations = {movie_id: recommended_movies[movie_id] / movie_weights[movie_id] for movie_id in recommended_movies if movie_weights[movie_id] != 0}
    

    top_recommendations = sorted(final_recommendations.items(), key=lambda x: x[1], reverse=True)[:rec_count]
    top_recommendations = [(movie_id, float(rating)) for movie_id, rating in top_recommendations]
    recommendations_df = pd.DataFrame(top_recommendations, columns=['movie_id', 'predicted_rating'])
    recommendation_spark_df = spark.createDataFrame(top_recommendations, ["movie_id", "predicted_rating"])
    recommendation_spark_df.show(truncate=False)

    return recommendations_df

print(recommend_movies(user_id=1, top_n=50, rec_count=10, user_movie_matrix=pivot_df, correlation_matrix=correlation_matrix))