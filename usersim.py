from pyspark import SparkConf, SparkContext
# from compute_simi import compute_item_similarity

# Initialize Spark configuration & context
conf = SparkConf().setAppName("UserBasedRecommendationSystem")
sc = SparkContext(conf=conf)

sc.setLogLevel("INFO") 
  
data = sc.textFile("ml-20m/ratings.csv")

header = data.first()
ratingsData = data.filter(lambda line: line != header).map(lambda line: line.split(","))

# Preprocess data to (userId, (movieId, rating))
ratings = ratingsData.map(lambda tokens: (int(tokens[0]), (int(tokens[1]), float(tokens[2]))))

userMoviePairs = ratings.join(ratings).filter(lambda x: x[1][0][0] < x[1][1][0])

# Prepare data for similarity computation: ((user1, user2), (rating1, rating2))
pairRatings = userMoviePairs.map(lambda x: ((x[1][0][0], x[1][1][0]), (x[1][0][1], x[1][1][1])))

groupedRatings = pairRatings.groupByKey().mapValues(list)

# Compute cosine similarity for each user pair
import math
def cosineSimilarity(ratingsPair):
    sum_xx, sum_yy, sum_xy = 0, 0, 0
    for ratingX, ratingY in ratingsPair:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
    return sum_xy / (math.sqrt(sum_xx) * math.sqrt(sum_yy))

similarities = groupedRatings.mapValues(cosineSimilarity)
targetUserId = 1

# N is top N similar users 
N = 10
topNSimilarUsers = similarities.filter(lambda x: x[0][0] == targetUserId or x[0][1] == targetUserId)\
                               .map(lambda x: (x[0][1], x[1]) if x[0][0] == targetUserId else (x[0][0], x[1]))\
                               .sortBy(lambda x: x[1], ascending=False)\
                               .take(N)

similarUserIds = [x[0] for x in topNSimilarUsers]

recommendedMovies = ratings.filter(lambda x: x[0] in similarUserIds)\
                           .map(lambda x: (x[1][0], x[1][1]))\
                           .groupByKey()\
                           .mapValues(list)\
                           .sortBy(lambda x: -len(x[1]))\
                           .keys()\
                           .take(10)


print("Recommended Movies for User", targetUserId, ":", recommendedMovies)

# with open('samplehere', 'w') as file:
#     # Write the text to the file
#     file.write('flag')