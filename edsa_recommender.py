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
from pygments import highlight
import streamlit as st
st.set_page_config(page_title="Movie Recommender Engine", layout="wide")

# Data handling dependencies
import pandas as pd
import numpy as np

from PIL import Image
from yaml import unsafe_load_all  # Image Processing

# Custom Libraries
from utils.data_loader import (
    load_movie_titles, load_data_for_eda, get_genres, get_years
)
from utils.visuals_creator import (
    create_metrics, create_ratings_distribution, create_count_vs_average_ratings_per_movie, create_top_n_frequently_rated_movie, create_top_n_frequently_rating_users, create_movies_production_over_time
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
                
                > A classic problem is that of millennials encountered today towards finding a good movie for them watch over the weekend without having to do too much research. Let’s see how we can solve this problem by helping them find a movie that they are most likely to enjoy through building a recommender application system in machine learning.
                
                The application uses the two most popular approaches for recommender systems:
                1. [Content-based filtering](https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/) and
                2. [Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering).
                
                #### 1. Content-based Filtering Approach
                Content-based recommendation systems work more closely with item features or attributes rather than user data. This approach recommends movies based on similarities among the contents of movies. They use features such as genre, director, cast, or description of movies a user react to to make recommendations based on these movie. 
                
                Similarity is the main key fundamental here, (i.e the most similar movie to what we have watched gets recommended to us) thus let's us look on how we computed the similarity.
                
                ##### Cosine Similarity: Computing similarity
                Since we are using textual data, we use the **consine similarity** to compute the similarity between the movies.
                
                We convert the textual data to a vector using ```TfifdVectorizer``` from ***sklearn's feature_extraction*** library creating the movies' features. A similar matrix is created with vectors that are used to check for the cosine angle between two movie vectors. This angle is used to show how similar this two movie vectors are to each other by comparing if the angle is 0. ***sklearn's*** ```cosine_similarity``` function is used here.
                
                Though content-based filtering doesn't need any data about users to make recommendations, a drawback is it tends to return on average items in a similar category with little variation across the recommendations.
                
                #### 2. Collaborative Filtering Approach
                Collaborative filtering works around the interactions that users have with items, finding patterns that the data about the items or users itself can’t.
                This approach recommends movies based on people's similar tastes and preferences liked in the past (i.e. historical data). In another word, this method predicts unknown ratings by using the similarities between users.
                
                The collaborative recommendation system is built using ```SVD``` model from ***scikit-surprise (surprise library)***
                
                ##### What is Singular Value Decomposition (SVD)?
                Singular Value Decomposition (SVD) and ``Matrix Factorization`` models are used to predict the end user's rating on items yet to be consumed by users.

                SVD is decomposition of a ```matrix R``` which is the utility matrix with ```m``` equal to the number of users and ```m``` number exposed movies ratings into the product of three matrices:

                - **U** is a left singular orthogonal matrix, representing the relationship between users and latent factors (Hopcroft & Kannan, 2012)

                - **Σ** is a diagonal matrix (with positive real values) describing the strength of each latent factor

                - **V(transpose)** is a right singular orthogonal matrix, indicating the similarity between items and latent factors.

                The general goal of SVD (and other matrix factorization methods) is to decompose the matrix ```R``` with all missing values and multiply its components, ```U```, ```Σ``` and ```V``` once again. As a result, there are no missing values and it is possible to recommend each user movies (items) they have not seen yet.
                
                Although collaborative filtering performs better than content based filtering in recommending items, it suffers from cold start problem for new items, data sparsity affecting the quality of the systems and scaling problems due to complexity of growing datasets.
                """)

    if page_selection == "EDA":
        # You may want to add more sections here for aspects such as an EDA,
        # or to provide your business pitch.
        st.title("Exploratory Data Analysis")

        df = load_data_for_eda("./resources/data")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")
        min_year, max_year = get_years(df.productionYear)
        # Add a year Slider to the Sidebar
        years = st.sidebar.slider(
            'Select Year Range', min_year, max_year, (min_year, max_year), 1)
        # Add a genres Multiselect to the Sidebar
        genres = st.sidebar.multiselect(
            "Select Genres", get_genres(df.genres), default=None)

        df = df[(df.productionYear >= years[0]) &
                (df.productionYear <= years[1])]

        if len(genres) > 0:
            visual_df = df.dropna(subset=['genres'])
            expression = '|'.join(
                [f"visual_df.genres.str.contains('{genre}', case=False, regex=False)" for genre in genres])
            visual_df = visual_df[eval(expression)]
        else:
            visual_df = df
        # Get metrics from the dataset
        movies, rated_movies, users, average_rating = create_metrics(
            visual_df[['movieId', 'userId', 'rating']])

        # Get the Visuals/Charts for the EDA
        # ---------------------------------------------
        # Create a count of ratings per rating histogram visual
        ratings_dist_viz = create_ratings_distribution(visual_df[['rating']])

        # Create a count of ratings vs average rating per movie scatter plot visual
        count_avg_rating_visual = create_count_vs_average_ratings_per_movie(
            visual_df[['title', 'rating']])

        # Create a count rating per movie to show the top 10 rated movies
        top_n_frequently_rated_movies = create_top_n_frequently_rated_movie(
            visual_df[['title', 'rating']], 10)

        # Create a count rating per user to show the top 10 rating users
        top_n_frequently_rating_users = create_top_n_frequently_rating_users(
            visual_df[['userId', 'rating']], 10)

        # Create a count of movies produced vs average rating over time
        movies_over_time = create_movies_production_over_time(
            visual_df[['productionYear', 'rating']])
        # Display the metrics
        metric1, metric2, metric3, metric4 = st.columns(4)
        with metric1:
            st.metric('Number of Movies', "{:,}".format(movies))
        with metric2:
            st.metric("Number of Rated Movies", "{:,}".format(rated_movies))
        with metric3:
            st.metric("Number of Users", "{:,}".format(users))
        with metric4:
            st.metric("Average Rating", average_rating)

        # Insert a break line to separate the metrics from other visuals
        st.write("---")
        # Display the visuals
        ratings_dist_viz_col, count_avg_rating_visual_col = st.columns(2)

        ratings_dist_viz_col.plotly_chart(
            ratings_dist_viz, use_container_width=True)
        count_avg_rating_visual_col.plotly_chart(
            count_avg_rating_visual, use_container_width=True)

        st.plotly_chart(top_n_frequently_rated_movies,
                        use_container_width=True)
        st.plotly_chart(top_n_frequently_rating_users,
                        use_container_width=True)
        st.plotly_chart(movies_over_time, use_container_width=True, height=600)
    # Create about page
    if page_selection == "About":
        # Creates a main title and subheader on your page -
        # these are static across all pages

        with st.container():
            st.markdown('''
                <div id="about-landing">
                <h1>Convert Your Data Into Dollars</h1>
                <div id="about-caption">To drive digital transformation, <span>DataLux Inc.</span> combines ML and AI to build rapidly deployable retail solutions for substantial revenue gains, finally making AI and ML solutions affordable for the retail industry.</div>
                </div>
                ''',
                        unsafe_allow_html=True)
        st.markdown('---')
        with st.container():
            highlight1, highlight2, highlight3, highlight4 = st.columns(4)
            highlight1.metric("Product Churn", 0)
            highlight2.metric("Bottom Line Growth", "$500M")
            highlight3.metric("Hours Saved", "10,000+")
            highlight4.metric("Integrations", "30+")
        with st.container():
            left1, right1 = st.columns(2)
            left1.subheader("Who we are?")
            left1.markdown('''
                We are the fastest-growing Retail, CPG and Supply Chain focused enterprise AI SaaS product company. We have built award winning products and our success is driven by focus on innovation and changing existing processes through automation.

                We have customers ranging from small business owners, SMEs to MMEs. Our SaaS solutions are built to provide quick visibility into your business, based on real time information, and enable smarter decisions based on data driven insights, while optimizing costs and adding millions back into the bottom line.
            ''')
            right1.image(Image.open("./resources/imgs/who-we-are.png"),
                         use_column_width="always")
        st.markdown("---")
        st.subheader(
            "To leverage digital and traditional data towards a competitive advantage",
            anchor="mission-statement")
        st.markdown("---")

        with st.container():
            left2, right2 = st.columns(2)
            left2.image(Image.open("./resources/imgs/why-us.jpg"),
                        use_column_width="always")
            right2.subheader("Why us?")
            right2.markdown('''
                Our team is comprised of retailer experts, management consultants, data scientists and product engineers. Together, we work in an agile environment to create products that work for every user and we continue to be committed to our users and foster long-term partnerships based on delivering high quality solutions.
            ''')

        st.markdown("---")

        # Make a list with the info (Name, destination, image_path) for the team members
        team_members = [
            ('CHIBUIKEM EFUGHA', 'Member', 'resources/imgs/CHIBUIKEM_EFUGHA.jpeg'),
            ('MICHAEL MWENDA', 'Member', 'resources/imgs/MICHAEL_MWENDA.jpg'),
            ('PATRICK ONDUTO', 'Member', 'resources/imgs/PATRICK_ONDUTO.jpg'),
            ('TOCHUKWU EZEOKAFOR', 'Member', 'resources/imgs/TOCHUKWU_EZEOKAFOR.jpeg'),
            ('ABIOLA AKINWALE', 'Member', 'resources/imgs/Placeholder.png'),
            ('SEYI ROTIMI', 'Member', 'resources/imgs/Placeholder.png'),
        ]

        _, meet_team, _ = st.columns(3)
        meet_team.header("Meet the Team")
        # Create a container to hold the  2row by 3 column grid
        with st.container():
            col0, col1, col2 = st.columns(3)
            col3, col4, col5 = st.columns(3)

        # Display the information of each member on a column
        for col, member in zip([col0, col1, col2, col3, col4, col5], team_members):
            name, destination, img_path = member
            with col:
                st.image(Image.open(img_path))
                st.header(name.title())
                st.write(destination)


with open('./resources/css/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
