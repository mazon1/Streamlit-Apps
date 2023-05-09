import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv("supermarket_sales.csv")

# Dashboard Page
def dashboard():
    st.title("Dashboard")
    st.write(''' The Supermarket Sales Prediction App is a machine learning application that allows users to explore and analyze the Supermarket Sales dataset. The app is divided into three main sections:

Dashboard: Provides a general overview of the dataset, including information such as the number of observations, mean and median values of key variables, and a breakdown of the product line categories.

EDA: Allows users to explore the relationships between different variables and how they impact sales. Users can select specific variables of interest and generate visualizations, such as scatterplots and heatmaps, to better understand these relationships.

Run Models: Enables users to build and test machine learning models to predict supermarket sales. Users can select the type of model they want to run (e.g. regression or classification), the input and output variables, and view the results of the model on a variety of metrics, such as R-squared and mean squared error.

Overall, the Supermarket Sales Prediction App provides a powerful tool for data exploration and machine learning modeling, allowing users to gain insights and make predictions about supermarket sales based on a variety of factors.''')
    
    # Display some general information about the dataset
    st.write("Number of rows:", len(data))
    st.write("Number of columns:", len(data.columns))
    st.write("Data types:", data.dtypes)
    st.write("Summary statistics:")
    st.write(data.describe())
    
    # Display some visualizations
    st.write("Histogram of ratings:")
    fig, ax = plt.subplots()
    sns.histplot(data=data, x="Rating", kde=True, ax=ax)
    st.pyplot(fig)
    
    st.write("Box plot of unit price by gender:")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x="Gender", y="Unit price", ax=ax)
    st.pyplot(fig)

# EDA Page
def eda():
    st.title("EDA")
    st.write("This page allows you to explore the dataset.")
    
    # Display some interactive visualizations
    st.write("Scatter plot of unit price vs. total by branch:")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="Unit price", y="Total", hue="Branch", ax=ax)
    st.pyplot(fig)

    # Correlation Plot
    st.header('Correlation Matrix')
    corr = data.select_dtypes(include=['float64', 'int']).corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(20, 10))
    sns.heatmap(corr, mask=mask, annot=True, center=0, cmap='coolwarm')
    st.pyplot(plt.show())

# Run Models Page
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Convert 'Date' to a datetime object
data['Date'] = pd.to_datetime(data['Date'])

# Date features
data['day_of_week'] = data['Date'].dt.dayofweek
data['day_of_month'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year

# Time features
data['Time'] = pd.to_datetime(data['Time'])
def map_time_interval(time):
    hour = time.hour
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

    # Apply the function to the 'Time' column
data['time_interval'] = data['Time'].apply(map_time_interval)

# One-Hot Encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=['City', 'Customer type', 'Gender', 'Product line', 'time_interval'], drop_first=True)

# Drop unnecessary columns
data_encoded.drop(['Invoice ID', 'Date', 'Time','Tax 5%','gross margin percentage','cogs','year'], axis=1, inplace=True)

# Identify non-numeric columns
non_numeric_columns = data_encoded.select_dtypes(include=['object']).columns

# Perform One-Hot Encoding for the remaining non-numeric columns
data_encoded = pd.get_dummies(data_encoded, columns=non_numeric_columns, drop_first=True)

# Define target variable and features
X = data_encoded.drop('Rating', axis=1)
y = data_encoded['Rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model comparison

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(max_iter=1000)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    print(f"{name}: {r2}")

def run_models():
    st.title("Run Models")
    st.write("This page allows you to run some machine learning models on the dataset.")
    
    # Provide some options for the user to select the type of model to run
    model_type = st.selectbox("Select the model type:", ["Linear Regression", "Decision Tree", "Random Forest","SVR", "KNN", "Neural Network"])
    
    # Provide some options for the user to select the input and output variables
    input_vars = st.multiselect("Select the input variables:", data.columns.drop("Rating"))
    output_var = "Rating"
    
    # Display the results of the model
    st.write("results:", results)

    


    
# Main App
def main():
    st.set_page_config(page_title="Invoice Data Analysis App")
    st.sidebar.title("Navigation")
    pages = {
        "Dashboard": dashboard,
        "EDA": eda,
        "Run Models": run_models
    }
    page = st.sidebar.selectbox("Select a page:", list(pages.keys()))
    pages[page]()
    
if __name__ == "__main__":
    main()
