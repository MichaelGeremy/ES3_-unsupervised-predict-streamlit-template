from matplotlib.pyplot import axis, legend
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    return (movies, rated_movies, users, average_rating)


def create_ratings_distribution(dataframe):
    """Creates ratings visuals 

    Args:
        dataframe (Pandas DataFrame): A dataframe of movies with the movieId, userId, rating, genres as columns
    Returns:
        plotly express bar pot figure
    """
    df = dataframe.groupby(
        'rating')['rating'].count().reset_index(name='counts')
    fig = px.bar(
        df,
        x="rating",
        y="counts",
        title="<b>Frequency of Ratings</b>",
        labels={
            "rating": "Rating",
            "counts": "Count of Ratings"
        }
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False)
    )
    
    top_rating_categories = df.sort_values('counts', ascending=False).rating.to_list()
    return fig, top_rating_categories


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
        title="<b>Count of Ratings by Average Rating</b>",
        labels={
            "mean": "Average Rating",
            "count": "Count of Ratings"
        },
        hover_name="title"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    rating_skewness = df['mean'].skew(axis=0)
    if rating_skewness > 1:
        skewness = 'highly positively (right) skewed'
    elif rating_skewness > 0.5 and rating_skewness <= 1:
        skewness = 'moderately positively (right) skewed'
    elif rating_skewness >= -1 and rating_skewness <= -0.5:
        skewness = 'moderately negatively (left) skewed'
    elif rating_skewness < -1:
        skewness = 'highly negatively (left) skewed'
    else:
        skewness = 'fairly symmetrical'
    
    return fig, (rating_skewness,skewness)


def create_top_n_frequently_rated_movie(dataframe, top_n):
    """Creates ratings visuals for the top 10 movies

    Args:
        dataframe (Pandas DataFrame): A dataframe of movies with the movieId, userId, rating, genres as columns
        top_n (int): An integer specify the top n movies to select
    Returns:
        plotly express bar plot figure
    """
    df = dataframe.dropna(subset=['rating']).groupby('title')['rating'].aggregate(
        ['count', 'mean']).sort_values('count', ascending=False).reset_index()
    df["mean"] = round(df["mean"], 1)
    top_rated_movies_df = df[:top_n]
    top_3_rated_movies = df[:3].values.tolist()
    top_rated_average = df['mean'].mean()
    top_rated_total_ratings = df['count'].sum()

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
    fig.update_layout(
        xaxis=dict(showgrid=False, type='category'),
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        legend_orientation="h",
        legend_title_side="top left"
    )
    return fig, top_3_rated_movies, top_rated_average, top_rated_total_ratings


def create_top_n_frequently_rating_users(dataframe, top_n):
    """Creates ratings visuals for the top 10 movies

    Args:
        dataframe (Pandas DataFrame): A dataframe of movies with the movieId, userId, rating, genres as columns
        top_n (int): An integer specify the top n movies to select
    Returns:
        plotly express bar plot figure
    """
    df = dataframe.dropna(subset=['rating']).groupby('userId')['rating'].aggregate(
        ['count', 'mean']).sort_values('count', ascending=False).reset_index()
    df["mean"] = round(df["mean"], 1)
    df["userId"] = df["userId"].astype("int32").astype("str")

    frequent_rating_users = df[:top_n]
    
    top_3_rating_users = frequent_rating_users[:3].values.tolist()
    top_user_ratings_average = frequent_rating_users['mean'].mean()
    top_users_ratings_total_ratings = frequent_rating_users['count'].sum()

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
    fig.update_layout(
        xaxis=dict(showgrid=False, type='category'),
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        legend_orientation="h",
        legend_title_side="top left"
    )
    return fig, top_3_rating_users, top_user_ratings_average, top_users_ratings_total_ratings


def create_movies_production_over_time(dataframe):
    """Creates a visual highlighting the number of movie produced per year
        And how they are rated by user (Average rating)

    Args:
        dataframe (Pandas DataFrame): A dataframe of movies with the rating, productionYear as columns
    Returns:
        plotly express bar and line plot figure
    """
    df = dataframe.dropna(subset=['productionYear']).groupby('productionYear')[['productionYear', 'rating']].aggregate(
        {'productionYear': 'count', 'rating': 'mean'}
    )
    df.columns = ['countYear', 'meanRating']
    df.reset_index(inplace=True)
    
    max_production_year = df.sort_values('countYear',ascending=False)[:1].values.tolist()
    min_rating_year = df.meanRating.min()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Bar(x=df.productionYear, y=df.meanRating,
               name="Average rating", opacity=0.5),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.productionYear, y=df.countYear,
                   name="Count of Produced Movies"),
        secondary_y=True,
    )
    # Update figure layout
    fig.update_layout(
        title_text="<b>Movies Production and Average Rating over Time</b>",
        yaxis=dict(title_text="Average Rating",
                   showgrid=False,
                   side='right'
                   ),
        yaxis2=dict(title_text="Count of Movies",
                    showgrid=False,
                    side='left'),
        xaxis=dict(title_text="Year of Production", type='category'),
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        legend_orientation="h",
        legend_title_side="top left"
    )
    return fig, max_production_year, min_rating_year
