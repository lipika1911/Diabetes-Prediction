import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
df = pd.read_csv('./diabetes.csv')

# Title and sidebar header
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')

# Display training data statistics
st.subheader('Training Data Stats')
st.write(df.describe())

# Split data into features (X) and target (y)
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to get user input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.0, 20.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_df = pd.DataFrame(user_report_data, index=[0])
    return report_df

# Get user input
user_data = user_report()

# Display user data
st.subheader('Patient Data')
st.write(user_data)

# Model training
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Predict user input
user_result = rf.predict(user_data)

# Visualizations
st.title('Visualised Patient Report')

# Color function
if user_result[0] == 0:
    color = 'blue'
else:
    color = 'red'

# Scatter plots
scatter_plots = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
plot_titles = ['Pregnancy count Graph (Others vs Yours)', 'Glucose Value Graph (Others vs Yours)',
               'Blood Pressure Value Graph (Others vs Yours)', 'Skin Thickness Value Graph (Others vs Yours)',
               'Insulin Value Graph (Others vs Yours)', 'BMI Value Graph (Others vs Yours)',
               'DPF Value Graph (Others vs Yours)', 'Age Value Graph (Others vs Yours)']

# Ensure scatter_plots and plot_titles have the same length
if len(scatter_plots) != len(plot_titles):
    st.error('Error: scatter_plots and plot_titles must have the same length.')
else:
    for i in range(len(scatter_plots)):
        fig = plt.figure()
        ax1 = sns.scatterplot(x='Age', y=scatter_plots[i], data=df, hue='Outcome', palette='Greens')
        ax2 = sns.scatterplot(x=user_data['Age'], y=user_data[scatter_plots[i]], s=150, color=color)  # Corrected access to user_data column
        plt.xticks(np.arange(10, 100, 5))
        plt.title('0 - Healthy & 1 - Unhealthy')
        st.header(plot_titles[i])
        st.pyplot(fig)

# Output
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)

# Accuracy
accuracy = accuracy_score(y_test, rf.predict(X_test)) * 100
st.subheader('Accuracy:')
st.write(f'{accuracy:.2f}%')
