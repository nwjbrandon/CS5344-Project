from pyspark.sql import SparkSession

def main():
    movies_df = spark.read.option("header", True).csv("ml-20m/movies.csv")
    movies_df.show()
    return

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Lab2").getOrCreate()
    main()