import pyspark.sql.functions as F

SIMILARITY_METHODS = ['cosine', ' adjusted_cosine', 'pearson']

def compute_item_similarity(ratings, user_col='userId', item_col='movieId', 
                            rating_col='rating', method='cosine', 
                            use_persist=False):
    """Compute item-item similarities from a dataframe with 
    user item ratings in long format.
    
    The dataframe should at have the following three columns:
    - user: a column with unique users ids
    - items: a column with unique item ids
    - rating: some rating the users gave to the item.
    
    The 'long format' refers to the fact that the algorithm is applied
    to (user, items, rating) tuples. And not to the wide
    (user, item_1_rating, item_2_rating, ...) tuples.
    
    Parameters
    ----------
    ratings : Spark dataframe
        Dataframe containing user item ratings.
    user_col : string, optional
        The user id column.  Default is 'user'.
    item_col : string, optional
        The item id column.  Default is 'item'.
    rating_col : string, optional
        The rating of the user for the item.  Default is 'rating'.
    method : string, optional
        The method for computing similarities. Default is 'cosine'.
    user_persist : boolean, optional
        Set to True for enabling persisting of dataframe(s) during computation.
    
    Returns
    -------
        Spark dataframe : item-item similarities
    """
    if method not in [SIMILARITY_METHODS]:
        raise ValueError("the similarity method {} is not supported".format(method)) 
    
    # rename relevant columns to user, item and rating
    user_item_ratings = (ratings
        .withColumnRenamed(user_col, 'user')
        .withColumnRenamed(item_col, 'item')
        .withColumnRenamed(rating_col, 'rating'))
    
    # persist dataframe if selected
    if use_persist:
        user_item_ratings.persist()
    
    if method=='cosine':
        # regular cosine similarity
        item_similarity = item_cosine_similarity(user_item_ratings)
        
        # unpersist data
        if use_persist:
            user_item_ratings.unpersist()
            
    elif method in ['adjusted_cosine', ' pearson']:
        # substract user mean for adjusted cosine
        # substract item mean for pearson r
        groupby_col = 'user' if method=='adjusted_cosine' else 'item'
        mean_rating = (user_item_ratings
            .groupby(groupby_col)
            .agg(F.mean(F.col('rating')).alias('mean_rating')))
        
        user_item_ratings_normed = (user_item_ratings
            .join(mean_rating, on=groupby_col, how='inner')
            .withColumn('rating', F.col('rating')-F.col('mean_rating'))
            .select('user','item','rating'))
        
        # redo persist of new normalized user item ratings
        if use_persist:
            user_item_ratings.unpersist()
            user_item_ratings_normed.persist()
        
        # compute cosine similarity on normed data
        item_similarity = item_cosine_similarity(user_item_ratings)
        
        # unpersist data
        if use_persist:
            user_item_ratings_normed.unpersist()        

    return item_similarity
    
def item_cosine_similarity(user_item_ratings):
    """Compute cosine similarities from a 
    dataframe with user item ratings in long format.
    
    Parameters
    ----------
    user_item_ratings : Spark dataframe
        Dataframe containing user item ratings.
        
    Returns
    -------
        Spark dataframe : item-item cosine similarities
    """
    # will compute similarity between items
    # and all other items
    other_items = (user_item_ratings
        .select(F.col('user').alias('user_other'),
                F.col('item').alias('item_other'),
                F.col('rating').alias('rating_other')))
    
    # cosine similarity numerator
    # \sum_i A_i*B_i
    numerator = (user_item_ratings
        .join(other_items, on=F.col('user')==F.col('user_other'), how='inner')
        .filter(F.col('item')!=F.col('item_other'))
        .groupby(['item','item_other'])
        .agg(F.sum(F.col('rating_other')*F.col('rating')).alias('cum_prod'))
        .select(F.col('item').alias('item_n'),
                F.col('item_other').alias('item_other_n'),
                F.col('cum_prod')))
    
    # item, item_other pairs for which numerator is not 0
    numerator_items = numerator.select(['item_n','item_other_n'])
    
    # compute the norms for all items
    # (\sum_i A_i^2)^{1/2}
    norms = (user_item_ratings
        .groupby('item')
        .agg(F.sqrt(F.sum(F.col('rating')*F.col('rating'))).alias('norm'))
        .select(F.col('item').alias('item_d'), F.col('norm')))
    
    # create other norms df
    norms_other = (norms
        .select(F.col('item_d').alias('item_other_d'),
                F.col('norm').alias('norm_other')))
    
    # compute cosine similarity denominator
    # (\sum_i A_i^2)^{1/2} * (\sum_i B_i^2)^{1/2}
    denominator = (numerator_items
        .join(norms, on=F.col('item_d')==F.col('item_n'), how='inner')
        .join(norms_other, on=F.col('item_other_d')==F.col('item_other_n'), how='inner')
        .withColumn('norm_prod', F.col('norm')*F.col('norm_other'))
        .select('item_d', 'item_other_d', 'norm_prod'))
    
    # combine numerator/denominator 
    # and compute the cosine similarity
    cosine_similarity = (denominator
        .join(numerator, on=[F.col('item_d')==F.col('item_n'),
                             F.col('item_other_d')==F.col('item_other_n')])
        .select(F.col('item_d').alias('item'), 
                F.col('item_other_d').alias('item_other'),
                (F.col('cum_prod')/F.col('norm_prod')).alias('sim')))
    
    return cosine_similarity