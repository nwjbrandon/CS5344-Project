import argparse

import numpy as np
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, udf
from pyspark.sql.types import FloatType

from dataset import MovieLens20m


def get_genres_dataset(spark):
    movieLens20m = MovieLens20m(spark)
    movies = movieLens20m.get_movies_df()
    movies = movies.withColumn("genres", split(col("genres"), "\\|"))

    return movies


def cosine_similarity_udf(movie_vector):
    # Access the broadcasted user vector within the UDF
    user_vector = np.array(user_vector_broadcast.value.toArray())
    movie_vector = np.array(movie_vector.toArray())
    return float(np.dot(movie_vector, user_vector) / (np.linalg.norm(movie_vector) * np.linalg.norm(user_vector)))


def get_user_input(movies):
    # Extracting the unique genres to choose from
    flat_genres = movies.withColumn("genre", explode("genres"))
    unique_genres = flat_genres.select("genre").distinct()
    genre_list = [row["genre"] for row in unique_genres.collect()]

    print("Please choose your favourite genres from the list below: ")
    for i, genre in enumerate(genre_list, start=1):
        print(f"{i}. {genre}")

    input_prompt = "Enter the numbers corresponding to your preferred genres, separated by commas (e.g., 1,3,5): "
    user_input = input(input_prompt)
    selected_indices = [int(index.strip()) for index in user_input.split(",")]
    selected_genres = [genre_list[index - 1] for index in selected_indices if 0 < index <= len(genre_list)]
    print(f"You selected: {selected_genres}")

    try:
        user_input = input(input_prompt)
        selected_indices = [int(index.strip()) for index in user_input.split(",") if index.strip().isdigit()]
        selected_indices = [index for index in selected_indices if 0 < index <= len(genre_list)]
        selected_genres = [genre_list[index - 1] for index in selected_indices]

        if not selected_genres:
            print("No valid genres selected. Please try again.")
            return get_user_input()

        return selected_genres

    except ValueError:
        print("Invalid input. Please enter the numbers corresponding to your preferred genres, separated by commas.")
        return get_user_input()


def setup_argparser():
    parser = argparse.ArgumentParser(description="Genre-Based Movie Recommender")
    parser.add_argument("-g", "--genres", type=str, help="Enter your preferred genres, separated by commas (e.g., Action,Adventure,Sci-Fi)", required=True)
    return parser


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CS5344 Project Initial Recommendation").getOrCreate()

    top_n_movies_to_recommend = 10

    movies = get_genres_dataset(spark=spark)

    # Applying CountVectorizer
    cv = CountVectorizer(inputCol="genres", outputCol="features")
    model = cv.fit(movies)
    movies_featured = model.transform(movies)

    # user_genres = ["Action", "Adventure", "Sci-Fi"]
    # user_genres = ["Crime", "War", "Horror"]
    # user_genres = get_user_input(movies)
    parser = setup_argparser()
    args = parser.parse_args()
    user_genres = args.genres.split(",")

    user_df = spark.createDataFrame([(user_genres,)], ["genres"])
    user_featured = model.transform(user_df)
    user_vector = user_featured.collect()[0]["features"]

    user_vector_broadcast = spark.sparkContext.broadcast(user_vector)

    cosine_similarity_udf = udf(cosine_similarity_udf, FloatType())

    # Calculate cosine similarity between user's genres and each movie's genres
    movies_featured = movies_featured.withColumn("similarity", cosine_similarity_udf(col("features")))
    top_movies = movies_featured.orderBy(col("similarity").desc()).limit(top_n_movies_to_recommend)
    top_movies.select("title", "genres", "similarity").show(truncate=False)

    spark.stop()
