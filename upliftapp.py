#conda activate chatgpt in terminal
import ssl
ssl._create_default_https_context= ssl._create_unverified_context
import pandas as pd

df = pd.read_csv('upliftfull.csv')

import streamlit as st

st.header('Exploratory Data Analysis ')

st.write(df.describe())

st.header('Dataframe')
st.dataframe(df)
selected_column = st.sidebar.selectbox('Select a column to visualize',df.columns)
#Seaborn Plot
import seaborn as sns

# Print the head of the dataset
print(df.head())

# Print the summary statistics of the numerical columns
print(df.describe())

# Plot the distribution of the "age" column.
#For some reason this sns.distplot was deprecated. Code may not run.
# sns.displot(df["Age"])

st.header('Histogram')
sns.histplot(df[selected_column])

st.header('Pyplot')
st.pyplot()
#Plotly Plot
import plotly.express as px

# # Plot the relationship between the "fare" and "age" columns. This opens in a different window
# fig = px.scatter(df, x="Fare", y="Age")
# fig.show()

st.write("Scatter plot")
x_axis = st.sidebar.selectbox('Select the x-axis',df.columns)
y_axis = st.sidebar.selectbox('Select the y-axis',df.columns)

fig = px.scatter(df, x=x_axis, y=y_axis)
st.plotly_chart(fig)


# st.write("Pair Plot")
# sns.pairplot(df, hue='class')
# st.pyplot()


st.header('Correlation plot')
corr = df.corr()
# selected_cols=df(['recency','history','treatment','target','used_bogo','used_discount'], inplace = True)
# corr = selected_cols.corr()
sns.heatmap(corr,annot=True)
st.pyplot()

st.header('Boxplot')

fig = px.box(df, y = selected_column)
st.plotly_chart(fig)