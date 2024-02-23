import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Define functions
def data_analysis_page(df_concat):
    # Function implementation...

# Main function
def main():
    # Load data and other initializations...
    df_concat = pd.read_csv('path_to_your_data.csv')

    # Call functions
    data_analysis_page(df_concat)

# Entry point of the script
if __name__ == "__main__":
    main()

 
def data_analysis_page(df_concat):
    st.title('Exploratory Data Analysis (EDA)')
 
    # Display sample data
    st.write('Sample Data:')
    st.write(df_concat.head())
 
    # Distribution of numerical features
    st.subheader('Distribution of Numerical Features')
    numerical_features = df_concat.select_dtypes(include='number').columns
    for feature in numerical_features:
        st.write(f'### {feature.capitalize()}')
        plt.figure(figsize=(8, 6))
        sns.histplot(df_concat[feature], kde=True)
        plt.title(f'Distribution of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Frequency')
        st.pyplot()
 
    # Correlation heatmap
    st.subheader('Correlation Heatmap')
    st.write(sns.heatmap(df_concat.corr(), annot=True, cmap='coolwarm'))
    st.pyplot()
 
    # Pairplot
    st.subheader('Pairplot')
    st.write(sns.pairplot(df, hue='Churn', palette='viridis'))
    st.pyplot()
 
    # Count plot for categorical features
    st.subheader('Count Plot for Categorical Features')
    categorical_features = df_concat.select_dtypes(include='object').columns
    for feature in categorical_features:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=feature, data=df_concat, hue='Churn', palette='viridis')
        plt.title(f'Count Plot of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Count')
        plt.legend(title='Churn', loc='upper right')
        st.pyplot()
 