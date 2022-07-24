import pandas as pd
import numpy as np
import plotly.express as px

VISUALS_COLOR = '#FF5714'


def create_metrics(dataframe):
    """Gets the number of movies, number of rated movies, number of users who rates those movies and average ratings of the movies


    Args:
        dataframe (Pandas Dataframe): A dataframe of movies with the movieId, userId, rating as columns

    Returns:
        List (numerical): A List in the form of [movies, rate_movies, users, avegare_rating]
    """
    movies = dataframe.movieId.nunique()
    rated_movies = dataframe.dropna(subset=['userId']).movieId.nunique()
    users = dataframe.userId.nunique()
    average_rating = round(dataframe.rating.mean(), 1)
    return [movies, rated_movies, users, average_rating]


def create_ratings_distribution(dataframe):
    """Creates ratings visuals 

    Args:
        dataframe (Pandas DataFrame): A dataframe of movies with the movieId, userId, rating, genres as columns
    Returns:
        plotly express bar pot figure
    """
    df = dataframe.groupby(
        'rating')['rating'].count().reset_index(name='count')
    fig = px.bar(
        df,
        x="rating",
        y="count",
        title="<b>Frequency of Ratings</b>",
        labels={
            "rating": "Rating",
            "count": "Count of Ratings"
        }
    )
    return fig


def create_count_vs_average_ratings_per_movie(dataframe):
    """Creates average ratings vs number of ratings per movie visuals 

    Args:
        dataframe (Pandas DataFrame): A dataframe of movies with the movieId, userId, rating, genres as columns
    Returns:
        plotly express scatter plot figure
    """
    df = dataframe.dropna(subset=['rating']).groupby('title')['rating'].aggregate(
        ['count', 'mean']).sort_values('count', ascending=False).reset_index()
    df["mean"] = round(df["mean"], 1)

    fig = px.scatter(
        df,
        x="mean",
        y="count",
        title="Count of Ratings by Average Rating ",
        labels={
            "mean": "Average Rating",
            "count": "Count of Ratings"
        },
        hover_name="title"
    )
    return fig


def create_top_n_frequently_rated_movie(dataframe, top_n):
    """Creates ratings visuals for the top 10 movies

    Args:
        dataframe (Pandas DataFrame): A dataframe of movies with the movieId, userId, rating, genres as columns
        top_n (int): An integer specify the top n movies to select
    Returns:

    """
    df = dataframe.dropna(subset=['rating']).groupby('title')['rating'].aggregate(
        ['count', 'mean']).sort_values('count', ascending=False).reset_index()
    df["mean"] = round(df["mean"], 1)
    top_rated_movies_df = df[:top_n]

    fig = px.bar(
        top_rated_movies_df.sort_values('count'),
        orientation="h",
        y="title",
        x="count",
        color="mean",
        title=f"<b>Top {top_n} Frequently Rated Movies</b>",
        labels={
            "title": "Movie Title",
            "count": "Number of Ratings",
            "mean": "Average Rating"
        }
    )
    fig.update_xaxes(type='category')
    return fig


def create_top_n_frequently_rating_users(dataframe, top_n):
    """Creates ratings visuals for the top 10 movies

    Args:
        dataframe (Pandas DataFrame): A dataframe of movies with the movieId, userId, rating, genres as columns
        top_n (int): An integer specify the top n movies to select
    Returns:

    """
    df = dataframe.dropna(subset=['rating']).groupby('userId')['rating'].aggregate(
        ['count', 'mean']).sort_values('count', ascending=False).reset_index()
    df["mean"] = round(df["mean"], 1)
    df["userId"] = df["userId"].astype("int32").astype("str")

    frequent_rating_users = df[:top_n]

    fig = px.bar(
        frequent_rating_users.sort_values('count'),
        orientation="h",
        y="userId",
        x="count",
        color="mean",
        title=f"<b>Top {top_n} Frequently Rating Users</b>",
        labels={
            "userId": "User ID",
            "count": "Number of Ratings",
            "mean": "Average Rating"
        }
    )
    fig.update_xaxes(type='category')
    return fig
