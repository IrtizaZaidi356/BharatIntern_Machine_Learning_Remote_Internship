# Step No# 01: Import necessary libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st
# Ignore warnings 
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="House Price Prediction", page_icon=":bar_chart:", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>",unsafe_allow_html=True)

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
house_price_predict_app = st.container()


with header:
    st.title("Welcome to Machine Learning Project!!!") 
    st.header("\n\nProject No# 01: House Price Prediction")
    st.subheader("Remote Internship | Bharat Intern")
    st.subheader("\n\n \nPerform Task:")
    st.text("Develop a machine learning model for predicting house prices using Python,scikit-learn, and TensorFlow.")

with dataset:
    # Step No# 02: Load the Dataset:
    st.header("\n\nHouse Price Prediction Dataset:")
    # Load the dataset
    df = pd.read_csv('Housing.csv')

with features:
    st.write(df.head())
    # Step No# 03: Convert Object Columns to Numeric:
    le = LabelEncoder()
    df['mainroad'] = le.fit_transform(df['mainroad'])
    df['guestroom'] = le.fit_transform(df['guestroom'])
    df['basement'] = le.fit_transform(df['basement'])
    df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
    df['airconditioning'] = le.fit_transform(df['airconditioning'])
    df['prefarea'] = le.fit_transform(df['prefarea'])
    df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])
    
    # Step No# 04: Exploratory Data Analysis (EDA) & Visualization
    st.title('House Price Prediction EDA')

    cl1, cl2 = st.columns(2)
    with cl1:
        # Display basic statistics
        st.subheader('\n**Basic Statistics:**')
        st.write(df.describe())    
    
    # Data Cleaning and Missing Value Handling
    with cl2:
        
        # Identify and handle missing values
        missing_values = df.isnull().sum()
        # Create a Seaborn heatmap to visualize missing values
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", ax=ax)
        st.subheader('Missing Values Heatmap:')
        st.pyplot(fig)
        # data = df.dropna()  # Drop rows with missing values for simplicity
    
    cl3, cl4 = st.columns(2)
    with cl3:
        # Correlation matrix
        st.subheader('**Correlation Matrix:**')
        correlation_matrix = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

    with cl4:
        # Pairplot for selected features
        st.subheader('**Pairplot for Selected Features:**')
        fig = px.scatter_matrix(df[['price', 'area', 'bedrooms', 'bathrooms']])
        st.plotly_chart(fig)
    
    cl5, cl6 = st.columns(2)
    # Bar Chart - Furnishing Status & Price
    with cl5:
        st.subheader('Bar Chart - Furnishing Status and Price:')
        fig_bar = px.bar(df, x='furnishingstatus', y="price", title='Furnishing Status & Price Distribution:')
        st.plotly_chart(fig_bar)
    
    # Box Chart - Bedrooms vs. Price
    with cl6:
        st.subheader('Box Chart - Bedrooms vs. Price:')
        fig_box = px.box(df, x='bedrooms', y='price', title='Bedrooms vs. Price')
        st.plotly_chart(fig_box)

    cl7, cl8 = st.columns(2)
    
    # Histogram - Price Distribution
    with cl7:
        st.subheader("Histogram - Price Distribution:")
        fig_hist = px.histogram(df, x='price', title='Price Distribution')
        st.plotly_chart(fig_hist)

    # Scatter Plot - Area vs. Price
    with cl8:
        st.subheader('Scatter Plot - Area vs. Price:')
        fig_scatter = px.scatter(df, x='area', y='price', title='Area vs. Price')
        st.plotly_chart(fig_scatter)

    # Bubble Chart - Bedrooms, Bathrooms (Size represents Price)
    st.subheader('Bubble Chart - Bedrooms vs Bathrooms (Size represents Price):')
    fig_bubble = px.scatter(df, x='bedrooms', y='bathrooms', size='price', title='Bedrooms, Bathrooms, and Price')
    st.plotly_chart(fig_bubble)

    # Bubble Chart - Area vs. Price (Size represents Bedrooms)
    st.subheader('Bubble Chart: Area vs. Price (Size represents Bedrooms)')
    fig_bubble = px.scatter(df, x='area', y='price', size='bedrooms', title='Bedrooms, Bathrooms, and Price')
    st.plotly_chart(fig_bubble)

    # Area Plot - Price vs. Stories
    st.subheader('Area Plot - Price vs. Stories:')
    fig_area = px.area(df, x='price', y='stories', title='Price vs. Stories')
    st.plotly_chart(fig_area)

    # Pie Chart - Air Conditioning
    st.subheader('Pie Chart - Air Conditioning:')
    fig_pie = px.pie(df, names='airconditioning', title='Air Conditioning Distribution')
    st.plotly_chart(fig_pie)

    st.subheader('Pie Chart - Furnishing Status:')
    fig_pie = px.pie(df, names='furnishingstatus', title='Furnishing Status Distribution')
    st.plotly_chart(fig_pie)


    # Multiple Charts - Bedrooms and Bathrooms Distribution
    st.subheader('Multiple Charts - Bedrooms and Bathrooms Distribution:')
    fig_multi = make_subplots(rows=1, cols=2, subplot_titles=('Bedrooms Distribution', 'Bathrooms Distribution'))
    fig_multi.add_trace(go.Histogram(x=df['bedrooms'], nbinsx=10, name='Bedrooms'), row=1, col=1)
    fig_multi.add_trace(go.Histogram(x=df['bathrooms'], nbinsx=10, name='Bathrooms'), row=1, col=2)
    fig_multi.update_layout(showlegend=False)
    st.plotly_chart(fig_multi)

with model_training:
    # Step 5: Train a Machine Learning Model using Sci-kit Learn
    #Train Test Split
    # Features and target variable
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression Model:
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model Evalution:
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 6: Train a Machine Learning Model using TensorFlow and Keras:
    # Build a Neural Network:
    model_tf = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    model_tf.compile(optimizer='adam', loss='mean_squared_error')

    # Train the Model:
    model_tf.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Streamlit App

with house_price_predict_app:
    # Step No# 07: User Input and Prediction with Streamlit

    # User Input Section
    st.sidebar.header('Enter House Details:')
    area = st.sidebar.slider('Area', min_value=500, max_value=20000, value=2000)
    bedrooms = st.sidebar.slider('Bedrooms', min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.slider('Bathrooms', min_value=1, max_value=5, value=2)
    stories = st.sidebar.slider('Stories', min_value=1, max_value=4, value=2)
    mainroad = st.sidebar.radio('Main Road', ['Yes', 'No'])
    guestroom = st.sidebar.radio('Guest Room', ['Yes', 'No'])
    basement = st.sidebar.radio('Basement', ['Yes', 'No'])
    hotwaterheating = st.sidebar.radio('Hot Water Heating', ['Yes', 'No'])
    airconditioning = st.sidebar.radio('Air Conditioning', ['Yes', 'No'])
    parking = st.sidebar.slider('Parking Spaces', min_value=0, max_value=4, value=2)
    prefarea = st.sidebar.radio('Preferred Area', ['Yes', 'No'])
    furnishingstatus = st.sidebar.radio('Furnishing Status', ['Furnished', 'Semi-Furnished', 'Unfurnished'])

    # Convert user input to DataFrame
    user_input = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [1 if mainroad == 'Yes' else 0],
        'guestroom': [1 if guestroom == 'Yes' else 0],
        'basement': [1 if basement == 'Yes' else 0],
        'hotwaterheating': [1 if hotwaterheating == 'Yes' else 0],
        'airconditioning': [1 if airconditioning == 'Yes' else 0],
        'parking': [parking],
        'prefarea': [1 if prefarea == 'Yes' else 0],
        'furnishingstatus': [0 if furnishingstatus == 'Unfurnished' else 1 if furnishingstatus == 'Semi-Furnished' else 2],
    })

    # Predict using scikit-learn model
    scikit_pred = model.predict(user_input)
    st.title("\n\nHouse Price Prediction:\n\n\n\n")
    st.subheader(f'Scikit-learn - Linear Regression Model Prediction: {round(scikit_pred[0])} $')

    # Predict using TensorFlow model
    tf_pred = model_tf.predict(user_input)
    st.title("\n\n\n\n")
    st.subheader(f'TensorFlow Model Prediction: {round(tf_pred[0][0])} $')
    