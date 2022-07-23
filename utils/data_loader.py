"""

    Helper functions for data loading and manipulation.

    Author: Explore Data Science Academy.

"""
# Data handling dependencies
import pandas as pd
import numpy as np


def load_movie_titles(path_to_movies):
    """Load movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Movie titles.

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_list = df['title'].to_list()
    return movie_list


def load_data_for_eda(path_to_movies):
    """Load movie and rating datasets for eda.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path folder containing the movie and ratings
        in .csv format.
    Returns
    -------
    Pandas Dataframe:
        movie ratings
    """

    movies = pd.read_csv(f'{path_to_movies}/movies.csv')
    ratings = pd.read_csv(f'{path_to_movies}/ratings.csv')

    df = movies.merge(ratings, how='outer', on='movieId')

    return df


def get_genres(series_with_genres):
    """Load Pandas Series to extract genres from a dataset

    Parameters
    ----------
    series_with_genres [Pandas Series]:
        Pandas series to extract with genres column
    Returns
    -------
    List[str]:
        genres
    """
    genres = series_with_genres[series_with_genres != '(no genres listed)']
    genres = genres.str.split('|').dropna().explode()
    return genres.unique().tolist()
