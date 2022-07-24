"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
# st.set_page_config(page_title="Movie Recommender Engine", layout="wide")

# Data handling dependencies
import pandas as pd
import numpy as np

from PIL import Image  # Image Processing

# Custom Libraries
from utils.data_loader import load_movie_titles, load_data_for_eda, get_genres
from utils.visuals_creator import (
    create_metrics, create_ratings_distribution, create_count_vs_average_ratings_per_movie, create_top_n_frequently_rated_movie, create_top_n_frequently_rating_users
)
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration


def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Solution Overview", "EDA", "About"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png', use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option', title_list[14930:15200])
        movie_2 = st.selectbox('Second Option', title_list[25055:25255])
        movie_3 = st.selectbox('Third Option', title_list[21100:21200])
        fav_movies = [movie_1, movie_2, movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                # try:
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = content_model(movie_list=fav_movies,
                                                        top_n=10)
                st.title("We think you'll like:")
                for i, j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)
                # except:
                #     st.error("Oops! Looks like this algorithm does't work.\
                #               We'll need to fix it!")

        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("""
                 
                [Recommender systems](https://en.wikipedia.org/wiki/Recommender_system) are one of the most common used applications of data science. Though much work has been done on this topic, the interest and demand in this area remains very high due to the rapid growth of the internet and information overload. It has become necessary for online businesses to help users to deal with information overload and provide personalized recommendations, content and services to them.
                
                The Recommender System seek to predict or filter preferences according to the user’s choices and this application recommends movies choices to movie lovers.
                
                The application uses the two most popular approaches for recommender systems:
                1. [Content-based filtering](https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/).
                2. [Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) and
                
                #### 1. Content-based Filtering Approach
                
                #### 2. Collaborative Filtering Approach
                This approach recommended items to a user that people with similar tastes and preferences liked in the past. In another word, this method predicts unknown ratings by using the similarities between users.
                
                The collaborative recommendation system is built using ```SVD``` model from ***scikit-surprise (surprise library)***
                
                ##### How does Singular Value Decomposition (SVD)?
                Singular Value Decomposition (SVD) and ``Matrix Factorization`` models are used to predict the end user's rating on items yet to be consumed by users.

                SVD is decomposition of a ```matrix R``` which is the utility matrix with ```m```` equal to the number of users and ```m``` number exposed movies ratings into the product of three matrices:

                - **U** is a left singular orthogonal matrix, representing the relationship between users and latent factors (Hopcroft & Kannan, 2012)

                - **Σ** is a diagonal matrix (with positive real values) describing the strength of each latent factor

                - **V(transpose)** is a right singular orthogonal matrix, indicating the similarity between items and latent factors.

                The general goal of SVD (and other matrix factorization methods) is to decompose the matrix ```R``` with all missing values and multiply its components, ```U```, ```Σ``` and ```V``` once again. As a result, there are no missing values and it is possible to recommend each user movies (items) they have not seen yet.
                """)
    if page_selection == "EDA":
        # You may want to add more sections here for aspects such as an EDA,
        # or to provide your business pitch.
        st.title("Exploratory Data Analysis")

        df = load_data_for_eda("./resources/data")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")
        genres = st.sidebar.multiselect(
            "Select Genres", get_genres(df.genres), default=None)

        if len(genres) == 0:
            visual_df = df
        else:
            visual_df = df.dropna(subset=['genres'])
            expression = '|'.join(
                [f"visual_df.genres.str.contains('{genre}', case=False, regex=False)" for genre in genres])
            visual_df = visual_df[eval(expression)]

        movies, rated_movies, users, average_rating = create_metrics(
            visual_df[['movieId', 'userId', 'rating']])

        # Create the metrics and display them
        metric1, metric2, metric3, metric4 = st.columns(4)
        with metric1:
            st.metric('Number of Movies', "{:,}".format(movies))
        with metric2:
            st.metric("Number of Rated Movies", "{:,}".format(rated_movies))
        with metric3:
            st.metric("Number of Users", "{:,}".format(users))
        with metric4:
            st.metric("Average Rating", average_rating)

        st.write("---")
        ratings_dist_viz = create_ratings_distribution(visual_df)
        ratings_dist_viz.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=False)
        )

        count_avg_rating_visual = create_count_vs_average_ratings_per_movie(
            visual_df)
        count_avg_rating_visual.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )

        top_n_frequently_rated_movies = create_top_n_frequently_rated_movie(
            visual_df, 10)
        top_n_frequently_rated_movies.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False)
        )

        top_n_frequently_rating_users = create_top_n_frequently_rating_users(
            visual_df, 10)
        top_n_frequently_rating_users.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False)
        )

        ratings_dist_viz_col, count_avg_rating_visual_col = st.columns(2)

        ratings_dist_viz_col.plotly_chart(
            ratings_dist_viz, use_container_width=True)
        count_avg_rating_visual_col.plotly_chart(
            count_avg_rating_visual, use_container_width=True)
        st.plotly_chart(top_n_frequently_rated_movies,
                        use_container_width=True)
        st.plotly_chart(top_n_frequently_rating_users,
                        use_container_width=True)

    # Create about page
    if page_selection == "About":
        # Creates a main title and subheader on your page -
        # these are static across all pages
        st.title("About Page")

        st.subheader("Project Team")

        # Make a list with the info (Name, destination, image_path) for the team members
        team_members = [
            ('CHIBUIKEM EFUGHA', 'Member', 'resources/imgs/Placeholder.png'),
            ('MICHAEL MWENDA', 'Member', 'resources/imgs/Placeholder.png'),
            ('PATRICK ONDUTO', 'Member', 'resources/imgs/Placeholder.png'),
            ('TOCHUKWU EFEOKAFOR', 'Member', 'resources/imgs/Placeholder.png'),
            ('ABIOLA AKINWALE', 'Member', 'resources/imgs/Placeholder.png'),
            ('SEYI ROTIMI', 'Member', 'resources/imgs/Placeholder.png'),
        ]

        # Create a container to hold the  2row by 3 column grid
        with st.container():
            col0, col1, col2 = st.columns(3)
            col3, col4, col5 = st.columns(3)

        # Display the information of each member on a column
        for col, member in zip([col0, col1, col2, col3, col4, col5], team_members):
            name, destination, img_path = member
            with col:
                st.image(Image.open(img_path))
                st.header(name)
                st.write(destination)


#------CSS STYLES------#
# Hide Streamlit Styles
hide_style = """
    <style>
        /*#MainMenu, footer, header {
            visibility:hidden;
        }*/
        .css-ocqkz7 img {
            width: 100%;
            aspect-ratio: 1/1;
            object-fit: cover;
            padding: 0.15rem;
            border-radius: 50%;
            border: 2px solid #0083B8;
        }

        .css-ocqkz7 .css-g5vx4n .css-1jikqva  h2 {
            padding: 0.25rem 0;
            font-size: 1.5rem;
            text-align: center;
        }

        .css-ocqkz7 .css-g5vx4n p {
            color: #FF5714;
            font-size: 1.15em;
            text-align: center;
        }
        
        .element-container.css-rztpzi > div {
            text_align: center;
            border-radius: 7px;
            padding: 1rem;
            box-shadow: 1px 1px 2px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #0083b8;
        }
        
        .element-container.css-rztpzi > div .css-1xarl3l{
            color: #00172b;
        }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
