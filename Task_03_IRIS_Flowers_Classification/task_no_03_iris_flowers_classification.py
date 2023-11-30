# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st
import os

# Ignore warnings 
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(page_title="IRIS Flowers Classification", page_icon=":sunflower:", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>",unsafe_allow_html=True)

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
iris_flowers_predict_app = st.container()


with header:
    st.title("Welcome to Machine Learning Project!!!") 
    st.header("\n\nProject No# 02: IRIS Flowers Classification")
    st.subheader("Remote Internship | Bharat Intern")
    st.subheader("\n\n \nPerform Task:")
    st.text("Develop a Machine Learning Model for classifying iris flowers based on their features using Python, scikit-learn, and TensorFlow.")

with dataset:
    # Step No# 02: Load the Dataset:
    st.header("\n\nIris Flowers Dataset:")
    # Load the dataset
    df = pd.read_csv('iris_flowers.csv')

with features:
    st.write(df.head())
    st.title('Exploratory Data Analysis (EDA):')

    
    # Display basic statistics in a table with increased width and height
    st.subheader('\n**Basic Statistics:**')
    # You can also adjust the layout width and height using the st.markdown function
    st.markdown("<style>div.Widget.row-widget.stTable {width: 700px !important}</style>", unsafe_allow_html=True)
    st.markdown("<style>div.Widget.col-widget.stTable {height: 400px !important}</style>", unsafe_allow_html=True)

    table = df.describe().transpose()
        
    table_styled = table.style.set_table_styles(
        [{'selector': 'th', 'props': [('font-size', '120%')]}]  # Increase font size of headers
    )

    # Streamlit table with custom styling
    st.table(table_styled)

    # You can also adjust the layout width and height using the `st.markdown` function
        
    cl1, cl2 = st.columns(2)
    with cl1:
        # Bar for selected features
        st.subheader('\n\n**Bar Chart: Species Distribution:**')
        fig_bar = px.bar([df['species'].value_counts()], title='Species Distribution:')
        fig_bar.update_layout(width=800, height=500)  # Adjust width and height as needed
        st.plotly_chart(fig_bar)
        
    # Data Cleaning and Missing Value Handling
    with cl2:
        
        # Identify and handle missing values
        missing_values = df.isnull().sum()
        # Create a Seaborn heatmap to visualize missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", ax=ax)
        st.subheader('Missing Values Heatmap:')
        st.pyplot(fig)    
    
    cl3, cl4 = st.columns(2)
    with cl3:
        # Correlation matrix
        st.subheader('\n\n**Correlation Matrix:**')
        # Exclude non-numeric columns from correlation analysis
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

    with cl4:
        # Multiple charts: Pair plot for pairwise relationships in the dataset
        st.subheader("Pair Plot: Pairwise Relationships:")
        sns.pairplot(df, hue='species')
        st.pyplot()

    cl5, cl6 = st.columns(2)
    
    with cl5:
        # Scatter plot for Sepal length vs Sepal width
        st.subheader('Scatter Plot: Sepal Length vs Sepal Width')
        fig, ax = plt.subplots()
        sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df, ax=ax)
        ax.set_title('Petal Length vs Petal Width')
        st.pyplot(fig)
    
    with cl6:
        st.subheader('Scatter Plot: Petal Length vs Petal Width')
        # Set up Matplotlib figure and axes
        fig, ax = plt.subplots()
        # Create scatter plot using Seaborn
        sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df, ax=ax)
        # Set plot title
        ax.set_title('Petal Length vs Petal Width')
        # Display the Matplotlib figure using st.pyplot()
        st.pyplot(fig)

with model_training:
    # Split the dataset into features (X) and target (y)
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = round(accuracy_score(y_test, y_pred) * 100,2)
    st.subheader("Random Forest Classifier:\n\n")
    st.write("\n\nAccuracy:", accuracy, "%")
    st.write("\n\nClassification Report:") 
    st.text(classification_report(y_test, y_pred))
    st.write("\n\nConfusion Matrix:") 
    st.text(confusion_matrix(y_test, y_pred))
    
    # Build a simple neural network
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    

    
    st.subheader("\n\n\nNeural Network with TensorFlow/Keras")
    model = Sequential()
    model.add(Dense(8, input_dim=X_scaled.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    st.write("Neural Network Accuracy:", accuracy)

with iris_flowers_predict_app:
    # Create a Streamlit app
    st.subheader("Iris Flowers Classification App")

    # Add user input for Sepal and Petal dimensions
    sepal_length = st.slider("Sepal Length", float(X['sepal_length'].min()), float(X['sepal_length'].max()), float(X['sepal_length'].mean()))
    sepal_width = st.slider("Sepal Width", float(X['sepal_width'].min()), float(X['sepal_width'].max()), float(X['sepal_width'].mean()))
    petal_length = st.slider("Petal Length", float(X['petal_length'].min()), float(X['petal_length'].max()), float(X['petal_length'].mean()))
    petal_width = st.slider("Petal Width", float(X['petal_width'].min()), float(X['petal_width'].max()), float(X['petal_width'].mean()))

    # Normalize user input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the trained Random Forest model
    rf_prediction = rf_model.predict(input_data_scaled)[0]
    nn_prediction = (model.predict(input_data_scaled) > 0.5).astype("int32")[0][0]

    # Display the predicted species
    st.subheader(f"Random Forest Predicted Species: {rf_prediction}")
    st.subheader(f"Neural Network Predicted Species: {nn_prediction}")