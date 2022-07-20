"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv', sep=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1, inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model = pickle.load(open('resources/models/SVD.pkl', 'rb'))


def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df, reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id, uid=ui, verbose=False))
    return predictions


def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store = []
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id=i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.


def collab_model(movie_list, top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    indices = pd.Series(movies_df['title'])

    movie_list = [movies_df[movies_df['title'] == i].movieId.values[0]
                  for i in movie_list]
    user_ids = pred_movies(movie_list)
    user_ids = list(set(user_ids))

    df_init_users = ratings_df[ratings_df['userId'] == user_ids[0]]
    for i in user_ids:
        df_init_users = df_init_users.append(
            ratings_df[ratings_df['userId'] == i])
    df_init_users.drop_duplicates(inplace=True)

    # Create a silimarity matrix between movies and users
    similarity = df_init_users.pivot_table(
        values='rating', index='userId', columns='movieId')
    # Normalize the matrix
    similarity_norm = similarity.subtract(similarity.mean(axis=1), axis=0)
    # Getting the cosine similarity matrix
    cosine_sim = cosine_similarity(similarity_norm.fillna(0).T)
    cosine_sim_df = pd.DataFrame(
        cosine_sim, columns=similarity.columns, index=similarity.columns)

    # Creating a Series with the similarity scores in descending order
    score_series = []
    for movie in movie_list:
        try:
            score_series.append(cosine_sim_df[movie])
        except KeyError:
            pass

    # Calculating the scores
    listings = pd.DataFrame(score_series[0])
    for i, score in enumerate(score_series):
        if i != 0:
            # Appending the names of movies
            listings.append(pd.DataFrame([score]))
    listings.columns = ['scores']

    # Choose top 50
    recommended_movies = []
    top_50_indexes = listings.sort_values(
        'scores', ascending=False).index[:50].to_list()

    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes, movie_list)

    # Get the title of the moves recommended
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df['title'])[i])

    return recommended_movies
