import ssl
ssl._create_default_https_context= ssl._create_unverified_context

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Add and resize an image to the top of the app
img_fuel = Image.open("../img/fuel_efficiency.png")
st.image(img_fuel, width=700)

st.markdown("<h1 style='text-align: center; color: black;'>Fuel Efficiency</h1>", unsafe_allow_html=True)

# Import train dataset to DataFrame
# Load the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = [
  'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
  'Acceleration', 'Model Year', 'Origin'
  ]

train_df = pd.read_csv(url, names=column_names, na_values='?', 
                      comment='\t', sep=' ', skipinitialspace=True)

# Create sidebar for user selection
with st.sidebar:
    # Available models for selection
    models = ["DNN", "TPOT"]

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        (models)
    )
    
    # Remove selected model 1 from model list
    # App refreshes with every selection change.
    models.remove(model1_select)
    
    model2_select = st.selectbox(
        "Choose Model 2:",
        (models)
    )

# Create tabs for separation of tasks
tab1, tab2 = st.tabs(["🗃 Data", "🔎 Model Results"])

with tab1:    
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(5))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)

with tab2:    
    
    # Columns for side-by-side model comparison
    col1, col2 = st.columns(2)

    # Build the confusion matrix for the first model.
    with col1:
        st.header(model1_select)

# Load the model

if model1_select == "DNN":
    # YOUR CODE GOES HERE!
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # Define the model architecture
    model1 = Sequential([
        Dense(64, activation='relu', input_shape=(7,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model1.compile(optimizer='adam',
                  loss='mse')

    # Train the model
    model1.fit(train_df.iloc[:, 1:8], train_df.iloc[:, 0], epochs=10, verbose=0)

elif model1_select == "TPOT":
    # YOUR CODE GOES HERE!
    from tpot import TPOTRegressor
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_df.iloc[:, 1:8], train_df.iloc[:, 0], test_size=0.2, random_state=42)

    # Define the TPOTRegressor pipeline
    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)

    # Fit the TPOTRegressor to the training data
    tpot.fit(X_train, y_train)

    # Evaluate the TPOTRegressor on the testing data
    tpot.score(X_test, y_test)

# Repeat the process for the second model
if model2_select == "DNN":
    # YOUR CODE GOES HERE!
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # Define the model architecture
    model2 = Sequential([
        Dense(64, activation='relu', input_shape=(7,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    #Compile the Model
    model2.compile(optimizer='adam',
              loss='mse')
    
    # Train the model
    model2.fit(train_df.iloc[:, 1:8], train_df.iloc[:, 0], epochs=10, verbose=0)
elif model2_select == "TPOT":
    from tpot import TPOTRegressor
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_df.iloc[:, 1:8], train_df.iloc[:, 0], test_size=0.2, random_state=42)
    # Define the TPOTRegressor pipeline
    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
    # Fit the TPOTRegressor to the training data
    tpot.fit(X_train, y_train)
    # Evaluate the TPOTRegressor on the testing data
    tpot.score(X_test, y_test)
    # Build the confusion matrix for the first model.
with col1:
    st.header(model1_select)
    if model1_select == "DNN":
        # Generate predictions
        y_pred1 = model1.predict(train_df.iloc[:, 1:8])

        # Build the confusion matrix
        cm1 = confusion_matrix(train_df.iloc[:, 0], y_pred1.round())

        # Display the confusion matrix
        cm1_fig = ff.create_annotated_heatmap(
            z=cm1,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['True Negative', 'True Positive'],
            colorscale='Viridis'
        )

        st.write(cm1_fig)

    elif model1_select == "TPOT":
        # Generate predictions
        y_pred1 = tpot.predict(train_df.iloc[:, 1:8])

        # Build the confusion matrix
        cm1 = confusion_matrix(train_df.iloc[:, 0], y_pred1.round())

        # Display the confusion matrix
        cm1_fig = ff.create_annotated_heatmap(
            z=cm1,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['True Negative', 'True Positive'],
            colorscale='Viridis'
        )

        st.write(cm1_fig)

        #Build the confusion matrix for the second model.
with col2:
    st.header(model2_select)
    if model2_select == "DNN":
        # Generate predictions
        y_pred2 = model2.predict(train_df.iloc[:, 1:8])

        # Build the confusion matrix
        cm2 = confusion_matrix(train_df.iloc[:, 0], y_pred2.round())

        # Display the confusion matrix
        cm2_fig = ff.create_annotated_heatmap(
            z=cm2,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['True Negative', 'True Positive'],
            colorscale='Viridis'
        )

        st.write(cm2_fig)

    elif model2_select == "TPOT":
        # Generate predictions
        y_pred2 = tpot.predict(train_df.iloc[:, 1:8])

        # Build the confusion matrix
        cm2 = confusion_matrix(train_df.iloc[:, 0], y_pred2.round())

        # Display the confusion matrix
        cm2_fig = ff.create_annotated_heatmap(
            z=cm2,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['True Negative', 'True Positive'],
            colorscale='Viridis'
        )

        st.write(cm2_fig)







