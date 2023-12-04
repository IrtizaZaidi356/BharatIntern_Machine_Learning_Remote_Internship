# Machine Learning | Remote Internship | Bharat Intern

## Perfom to Task No# 01: House Price Prediction | Machine Learning:
  - Develop a machine learning model for predicting house prices using Python, scikit-learn, and TensorFlow.
    
---

### **Project No# 01: House Price Prediction:**

#### **Introduction:**
 - The House Price Prediction project (Project No# 01) aims to develop a machine learning model using Python, scikit-learn, and TensorFlow for predicting house prices. The primary objective is to create a robust model capable of providing accurate price estimates based on various features like area, number of bedrooms, bathrooms, and other relevant factors. This project is part of a remote internship with Bharat Intern, focusing on practical applications of machine learning in real-world scenarios.

#### Understanding Objectives:
 #### 1) **Data Exploration and Cleaning:**
   - The project begins with importing essential libraries and loading the house price prediction dataset. The dataset, sourced from 'Housing.csv,' contains information on various features like area, bedrooms, bathrooms, etc. Object columns are converted to numeric using Label Encoding to make them compatible with machine learning algorithms.

 #### 2) **Exploratory Data Analysis (EDA):**
   - Extensive exploratory data analysis is conducted to understand the relationships between different features and the target variable (price). The project utilizes various visualizations such as bar charts, box plots, pair plots, correlation matrices, histograms, scatter plots, pie charts, bubble charts, and area plots to gain insights into the dataset's characteristics.

 #### 3) **Model Training:**
  - The machine learning models are trained using scikit-learn and TensorFlow. A linear regression model from scikit-learn is employed for its simplicity and interpretability. Additionally, a neural network using TensorFlow and Keras is developed to capture complex patterns in the data. The training process involves splitting the data into training and testing sets, feature scaling, and model fitting.

 #### 4) **Streamlit App:**
  - To enhance user interaction, a Streamlit web app is created. Users can input details such as area, bedrooms, bathrooms, and other features to obtain house price predictions. The app seamlessly integrates the trained scikit-learn and TensorFlow models, providing users with predictions in real-time.

#### **Methodologies:**
  - The project follows a structured approach, starting with data exploration and cleaning, moving to EDA, and culminating in model training and application development. The utilization of both traditional machine learning (scikit-learn) and deep learning (TensorFlow) showcases a comprehensive approach to solving the regression problem.

#### **Outcomes:**
  - **Data Understanding:** The EDA section reveals key insights into the dataset, highlighting relationships and patterns that contribute to house prices.
  - **Model Performance:** The scikit-learn Linear Regression model and TensorFlow neural network are trained to accurately predict house prices. Evaluation metrics such as mean squared error are used to assess model performance.
  - **Streamlit App:** The Streamlit web app provides a user-friendly interface for predicting house prices based on user input. It enhances accessibility and usability, showcasing the practical application of the developed models.

#### **Conclusion:**
  - The House Price Prediction project successfully demonstrates the application of machine learning in predicting real estate prices. The combination of scikit-learn and TensorFlow allows for a well-rounded approach, considering both traditional and deep learning methodologies. The Streamlit app further extends the project's impact by making the models accessible to a broader audience. Through this project, a foundation is laid for understanding, implementing, and deploying machine learning models in real-world scenarios, contributing to the broader field of data science and artificial intelligence.

---

## **Perform Task No# 02: Movie Recommendations | Machine Learning:
  - Build a Movie Recommendation System using collaborative filtering and machine learning techniques in Python

---

### **Project No# 02: Movie Recommendation System:**

 #### **Introduction:**
  - The objective of Project No# 02 was to build a Movie Recommendation System using collaborative filtering and machine learning techniques in Python. The project aimed to create a system that could analyze user preferences and provide personalized movie recommendations. The dataset consisted of two main CSV files: 'Netflix_Movie.csv' containing information about movies, and 'Netflix_Rating.csv' with user ratings for those movies.

 #### **Understanding Objectives:**
   1) **Data Loading and Cleaning:** The project began with importing necessary libraries and loading the movie and rating datasets. The datasets were merged based on the 'Movie_ID' column, resulting in a comprehensive dataset containing both movie information and user ratings.
   2) **Exploratory Data Analysis (EDA):** The EDA phase involved a thorough exploration of the dataset. Descriptive statistics were used to understand the central tendencies of the data, and various visualizations were employed to analyze the distribution of ratings across users, movie ratings over the years, and rating spreads for each movie. The EDA phase aimed to uncover patterns and insights within the dataset.
   3) **Handling Data Quality Issues:** The dataset was checked for missing values and duplicates. Missing values were visualized using a heatmap, and duplicates were removed to ensure data integrity.
   4) **Handling Outliers:** Outliers in the rating distribution were identified using boxplots, and appropriate techniques were applied to handle these outliers.
   5) **Data Preprocessing:** The dataset was split into training and testing sets. Standardization was performed using the StandardScaler from scikit-learn to ensure uniformity in the data.
   6) **Machine Learning Model (Random Forest):** A Random Forest Regressor was implemented for the machine learning aspect of the project. The model was trained on features such as 'Movie_ID,' 'Year,' and 'User_ID,' and predictions were made for the testing set. Mean Squared Error (MSE) was used as an evaluation metric.
   7) **Neural Network Model (TensorFlow/Keras):** A neural network was implemented using TensorFlow and Keras for a more complex recommendation system. The model architecture included embedding layers for categorical variables, a flattening layer, and dense layers. The model was trained and validated using the training set.
   8) **User Input for Recommendations:** A user-friendly interface was created to allow users to input a Movie ID and receive a predicted rating from the neural network model.
 
 #### **Methodologies and Outcomes:**
  - The project utilized a combination of exploratory data analysis, machine learning techniques (Random Forest), and neural networks to build a movie recommendation system. Visualizations provided insights into the distribution of ratings, trends over the years, and user preferences. The machine learning model (Random Forest) demonstrated its capability to predict ratings, while the neural network model added complexity for more sophisticated recommendations.

#### **Conclusion:**
  - Project No# 02 successfully achieved its objectives by creating a movie recommendation system that leverages collaborative filtering and machine learning. The project demonstrated the importance of data exploration, cleaning, and the application of diverse models to cater to different complexities. The user input feature allows for practical interaction with the recommendation system. This project serves as a foundation for further enhancements and optimizations in the field of recommendation systems, contributing to a more personalized and engaging user experience in the realm of movie recommendations.

---

## Perfom to Task No# 03:IRIS Flowers Classification | Machine Learning:
  - Develop a Machine Learning Model for classifying iris flowers based on their features using Python, scikit-learn, and TensorFlow.

---

### **Project No# 03: IRIS Flowers Classification:**

#### 1) **Introduction:**
  - Project No# 03 involves the development of a machine learning model for the classification of iris flowers based on their features. The primary goal is to leverage Python programming along with scikit-learn and TensorFlow libraries to create a robust and accurate model. The project is conducted as part of a remote internship at Bharat Intern, contributing to hands-on experience and skill development.

#### 2) **Understanding Objectives:**
  - The project aims to achieve the following objectives:

  - i) **Exploratory Data Analysis (EDA):** Understand the structure and characteristics of the iris flowers dataset. Utilize various visualizations, such as bar charts, box plots, pair plots, correlation matrices, histograms, scatter plots, pie charts, bubble charts, and area plots, to gain insights into the data distribution and relationships.

  - ii) **Data Pre-processing and Feature Engineering:** Handle any missing values, clean the data, and perform feature engineering if necessary. Standardize the data using the Standard Scaler to ensure consistent scales for the features.

  - iii) **Handling Outliers:** Use the Interquartile Range (IQR) method to identify and remove outliers from the dataset, ensuring a cleaner and more reliable input for model training.

  - iv) **Random Forest Classifier:** Implement a Random Forest Classifier using scikit-learn. Train the model on the pre-processed dataset, evaluate its accuracy, and provide a comprehensive classification report and confusion matrix.

  - v) **Neural Network with TensorFlow/Keras:** Build a neural network using TensorFlow and Keras. Train the model on the standardized dataset, evaluate its performance, and compare the results with the Random Forest Classifier.

  - vi) **Streamlit Application:** Develop a user-friendly Streamlit application that allows users to input values for sepal length, sepal width, petal length, and petal width. Display the predictions from both the Random Forest Classifier and Neural Network, offering an interactive and informative interface.

#### 3) **Methodologies:**
  - The project employs a systematic approach, beginning with data exploration and visualization to gain a deep understanding of the dataset. Data preprocessing involves handling missing values, cleaning the data, and applying feature engineering techniques. The use of the IQR method ensures the removal of outliers, enhancing the model's robustness.

  - Two machine learning models are implemented – the Random Forest Classifier from scikit-learn and a neural network using TensorFlow and Keras. Both models undergo training and evaluation, allowing for a comprehensive comparison of their performance.

  - The development of a Streamlit application provides an accessible interface for users to interact with the trained models, inputting values and receiving real-time predictions.

#### 4) **Outcomes:**
  - Upon completion of the project, a fully functional machine learning model for iris flowers classification is achieved. The outcomes include visually rich exploratory data analysis, effective data preprocessing, accurate model training, and the creation of an interactive Streamlit application.

#### 5) **Conclusion:**
  - Project No# 03 has successfully met its objectives, providing valuable insights into the application of machine learning techniques for iris flowers classification. The utilization of scikit-learn and TensorFlow showcases the versatility and power of these libraries in building robust and accurate models. The Streamlit application enhances user engagement and demonstrates the practicality of deploying machine learning solutions in real-world scenarios. This project contributes to the continuous learning and skill development of the intern, fostering a deeper understanding of machine learning methodologies and their applications.
