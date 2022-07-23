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

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles, load_data_for_eda, get_genres
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

from PIL import Image  # Image Processing
# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
st.set_page_config(page_title="Movie Recommender Engine", layout="wide")


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
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

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
        import matplotlib.pyplot as plt
        import seaborn as sns
        st.title("Exploratory Data Analysis")
        df = load_data_for_eda("./resources/data")
        st.sidebar()

        rated_movies = ratings.movieId.nunique()
        users = ratings.userId.nunique()
        average_rating = round(ratings.rating.mean(), 1)

        left_card, middle_card, right_card = st.columns(3)

        # Create the figures to display
        ratings_dist = sns.catplot(x="rating", data=movies_merged[[
                                   'rating']], kind='count', color="#001001", aspect=1.25)
        ratings_dist.set_ylabels("counts")
        ratings_dist.set(title="Total number of ratings")

        with left_card:
            st.subheader(rated_movies)
            st.write("Number of Rated Movies")

        with middle_card:
            st.subheader(users)
            st.write("Number of Users")

        with right_card:
            st.subheader(
                f"{average_rating} {':star:' * int(round(average_rating, 0))}")
            st.write("Average Movies Rating")

        st.write("---")
        st.pyplot(ratings_dist)

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
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
