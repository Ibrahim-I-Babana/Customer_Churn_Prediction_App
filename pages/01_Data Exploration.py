# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pyodbc
# import pickle

# st.set_page_config(
#     page_title='View Data',
#     page_icon=':)',
#     layout='wide'
# )

# st.title('Exploratory Analysis of Telecommunications Data (EDA)')

# #Display dashboard image
# st.image("Images/Global Telecommunications Network.jpg", caption="Telecommunications Customer Churn Exploration", use_column_width=True)

# # # Load the dataset
# # dataset_path = (r'C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\df_concat.csv')
# # df = pd.read_csv(dataset_path)




# # Initialize connection.
# def init_connection():
#     return pyodbc.connect(
#         "DRIVER={SQL Server};SERVER="
#         + st.secrets["SERVER"]
#         + ";DATABASE="
#         +  st.secrets["DATABASE"]
#         + ";UID="
#         +  st.secrets["USERNAME"]
#         + ";PWD="
#         + st.secrets["PASSWORD"]
#     )

# conn = init_connection()

# # Perform query.
# @st.cache_data
# def query_database(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         rows = cur.fetchall()
#         df = pd.DataFrame.from_records(data=rows, columns=[column[0] for column in cur.description])
        
#     return df

# # Select numeric features.
# @st.cache_data
# def select_numeric_features():
#     query = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
#     df = query_database(query)
#     numeric_cols = df.select_dtypes(include=np.number).columns
#     df_numeric = df[numeric_cols]
#     return df_numeric

# # Select all features.
# @st.cache_data
# def select_all_features(query):
#     df = query_database(query)
#     return df   

# # Function to plot count plot for categorical features.
# def plot_categorical_countplot(df):
#     st.subheader('Count Plot for Categorical Features')
#     categorical_features = df.select_dtypes(include='object').columns
#     for feature in categorical_features:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         sns.countplot(x=feature, data=df, palette='viridis', ax=ax)
#         ax.set_title(f'Count Plot of {feature.capitalize()}')
#         ax.set_xlabel(feature.capitalize())
#         ax.set_ylabel('Count')
#         st.pyplot(fig)

# # Function to plot correlation heatmap.
# def plot_correlation_heatmap(df):
#     st.subheader('Correlation Heatmap')
#     numeric_df = df.select_dtypes(include='number')
#     if not numeric_df.empty:
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
#         st.pyplot(fig)
#     else:
#         st.write("No numeric columns found to compute correlation.")

# # Function to plot distribution of numerical features.
# def plot_numerical_distribution(df):
#     st.subheader('Distribution of Numerical Features')
#     numerical_features = df.select_dtypes(include='number').columns
#     for feature in numerical_features:
#         st.write(f'### {feature.capitalize()}')
#         fig, ax = plt.subplots(figsize=(8, 6))
#         sns.histplot(df[feature], kde=True)
#         ax.set_title(f'Distribution of {feature.capitalize()}')
#         ax.set_xlabel(feature.capitalize())
#         ax.set_ylabel('Frequency')
#         st.pyplot(fig)

# # Function to plot pairplot.
# def plot_pairplot(df):
#     st.subheader('Pairplot')
#     fig = sns.pairplot(df, palette='viridis')
#     st.pyplot(fig)

# # Add the image
# st.image('C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Customer_Churn_Prediction_App/Images/Global Telecommunications Network.jpg', use_column_width=True)

# if __name__ == "__main__":
#     col1, col2 = st.columns(2)
#     with col1:
#         selected_option = st.selectbox('Select the type of feature', options=['All features', 'Numeric features'], key='selected_columns')
#     with col2:
#         pass
    
#     if selected_option == 'All features':
#         data = select_all_features("select * from dbo.LP2_Telco_churn_first_3000")
#     elif selected_option == 'Numeric features':
#         data = select_numeric_features()
    
#     st.dataframe(data)

#     # Perform analysis based on selected option
#     if selected_option == 'All features':
#         plot_categorical_countplot(data)
#         plot_correlation_heatmap(data)
#     elif selected_option == 'Numeric features':
#         plot_numerical_distribution(data)
#         plot_pairplot(data)

#     st.write(st.session_state)


# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pyodbc

# st.set_page_config(
#     page_title='View Data',
#     page_icon=':)',
#     layout='wide'
# )

# st.title('Exploratory Analysis of Telecommunications Data (EDA)')

# # Initialize connection.
# def init_connection():
#     return pyodbc.connect(
#         "DRIVER={SQL Server};SERVER="
#         + st.secrets["SERVER"]
#         + ";DATABASE="
#         +  st.secrets["DATABASE"]
#         + ";UID="
#         +  st.secrets["USERNAME"]
#         + ";PWD="
#         + st.secrets["PASSWORD"]
#     )

# conn = init_connection()

# # Perform query.
# @st.cache_resource
# def query_database(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         rows = cur.fetchall()
#         df = pd.DataFrame.from_records(data=rows, columns=[column[0] for column in cur.description])
        
#     return df

# # Function to display descriptive statistics.
# def display_descriptive_statistics(df):
#     st.subheader("Descriptive Statistics")
#     st.write("#### Summary Statistics:")
#     st.write(df.describe().T)
#     st.write("#### Summary Statistics for Categorical Columns:")
#     st.write(df.describe(include='object').T)
#     st.write("#### Sample Data:")
#     st.write(df.head())
#     st.write("#### Dataset Info:")
#     st.write(df.info())

# # Add the image
# st.image('Images/Global Telecommunications Network.jpg', caption="Telecommunications Customer Churn Exploration", use_column_width=True)

# # Load the dataset and display descriptive statistics
# query = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
# df_concat = query_database(query)
# display_descriptive_statistics(df_concat)

# # Additional analysis
# st.subheader("Additional Analysis")
# # Correlation
# correlation = df_concat.corr(numeric_only=True)
# st.write("#### Correlation Matrix:")
# st.write(correlation)

# # Churn counts by Internet Service
# churn_counts = df_concat.groupby(['InternetService', 'Churn']).size().unstack()
# st.write("#### Churn Counts by Internet Service:")
# st.write(churn_counts)



import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyodbc

st.set_page_config(
    page_title='View Data',
    page_icon=':)',
    layout='wide'
)

st.title('Exploratory Analysis of Telecommunications Data (EDA)')

# Initialize connection.
def init_connection():
    return pyodbc.connect(
        "DRIVER={SQL Server};SERVER="
        + st.secrets["SERVER"]
        + ";DATABASE="
        +  st.secrets["DATABASE"]
        + ";UID="
        +  st.secrets["USERNAME"]
        + ";PWD="
        + st.secrets["PASSWORD"]
    )

conn = init_connection()

# Perform query.
@st.cache_resource
def query_database(query):
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        df = pd.DataFrame.from_records(data=rows, columns=[column[0] for column in cur.description])
        
    return df

# Function to display descriptive statistics.
def display_descriptive_statistics(df):
    st.subheader("Descriptive Statistics")
    st.write("#### Summary Statistics:")
    st.write(df.describe().T)
    st.write("#### Summary Statistics for Categorical Columns:")
    st.write(df.describe(include='object').T)
    st.write("#### Sample Data:")
    st.write(df.head())
    st.write("#### Dataset Info:")
    st.write(df.info())

# # Add the image
# st.image('Images/Global Telecommunications Network.jpg', caption="Telecommunications Customer Churn Exploration", use_column_width=True)

# Load the dataset and display descriptive statistics
query = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
df_concat = query_database(query)
display_descriptive_statistics(df_concat)

# Additional analysis
st.subheader("Additional Analysis")
# Correlation
correlation = df_concat.corr(numeric_only=True)
st.write("#### Correlation Matrix:")
fig_corr = go.Figure(data=go.Heatmap(z=correlation.values,
                                     x=correlation.columns,
                                     y=correlation.columns,
                                     colorscale='Viridis'))
st.plotly_chart(fig_corr)

# Churn counts by Internet Service
#Churn counts by Internet Service
churn_counts = df_concat.groupby(['InternetService', 'Churn']).size().unstack()
st.write("#### Churn Counts by Internet Service:")
st.write(churn_counts)

churn_counts = df_concat.groupby(['InternetService', 'Churn']).size().unstack()
st.write("#### Churn Counts by Internet Service:")
fig_churn_counts = px.bar(churn_counts, x=churn_counts.index, y=churn_counts.columns,
                          title="Churn Counts by Internet Service", barmode='group')
st.plotly_chart(fig_churn_counts)



# import streamlit_authenticator
# import streamlit as st
# import os
# import yaml
# from streamlit_authenticator import Authenticate
# from Authenticate_Files.validator import Validator
# from Authenticate_Files.utils import generate_random_pw
# from session_state import SessionState
# import uuid
# from Authenticate_Files.hasher import Hasher

# # Function to generate unique key
# def generate_key():
#     return str(uuid.uuid4())

# # Load configuration from YAML file
# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=yaml.SafeLoader)

# # Set the favicon and title
# st.set_page_config(page_title="Telecommunications Customer Churn Prediction App", page_icon="🚀")

# # Initialize session state
# session_state = SessionState(authenticated=False, form_state={
#     "username": "",
#     "password": "",
#     "new_username": "",
#     "new_password": "",
#     "confirm_password": ""
# })

# # Home page function
# def home_page():
#     st.title('Welcome to Telecommunications Customer Churn Prediction App')
#     st.markdown("""
#     Losing customers is a major cost for any organization. Customer churn, also known as customer attrition or customer turnover, is the percentage of customers who stop using your company’s product or service within a certain period of time.
    
#     For example, if you started the year with 500 customers and ended with 480 customers, your customer churn rate would be 4%.
    
#     If we could understand the reasons and the timing of customer churn with reasonable accuracy, it would greatly help the organization to plan and implement effective retention strategies.

#     This is a Classification project aimed at assisting a telecommunications company understand their data and find the life time value of each customer and know what factors affect the rate at which customers stop using their network.
#     """)
    
#     # Add image
#     st.image("Images/Global Telecommunications Network.jpg", caption="Telecommunications Logo", use_column_width=True)

# # Logout function
# def logout():
#     session_state.authenticated = False

# # Login function
# def login():
#     username = session_state.form_state["username"]
#     password = session_state.form_state["password"]
#     # Dummy authentication (replace with your own authentication logic)
#     if username == "admin" and password == "password":
#         session_state.authenticated = True
#     else:
#         st.error("Login failed: Incorrect username or password.")

# # Create Account function
# def create_account():
#     new_username = session_state.form_state["new_username"]
#     new_password = session_state.form_state["new_password"]
#     confirm_password = session_state.form_state["confirm_password"]
#     # Dummy account creation (replace with your own account creation logic)
#     if new_password != confirm_password:
#         st.error("Passwords do not match.")
#     else:
#         st.success(f"Account for {new_username} created successfully!")

# # Main function
# def main():
#     # Login and create account form
#     st.title("User Authentication")
#     session_state.form_state["username"] = st.text_input("Username", key=f"login_username_input_{generate_key()}", value=session_state.form_state["username"])
#     session_state.form_state["password"] = st.text_input("Password", type="password", key=f"login_password_input_{generate_key()}", value=session_state.form_state["password"])
#     login_button_key = generate_key()
#     login_button = st.button("Login", key=login_button_key)
#     session_state.form_state["new_username"] = st.text_input("New Username", key=f"new_username_input_{generate_key()}", value=session_state.form_state["new_username"])
#     session_state.form_state["new_password"] = st.text_input("New Password", type="password", key=f"new_password_input_{generate_key()}", value=session_state.form_state["new_password"])
#     session_state.form_state["confirm_password"] = st.text_input("Confirm Password", type="password", key=f"confirm_password_input_{generate_key()}", value=session_state.form_state["confirm_password"])
#     create_account_button_key = generate_key()
#     create_account_button = st.button("Create Account", key=create_account_button_key)

#     if create_account_button:
#         create_account()

#     if login_button:
#         login()

#     # Display the home page content
#     home_page()

#     # How to run the app
#     st.title("How to Run the App")
#     st.markdown("To run this app locally, make sure you have Python installed. Then, install Streamlit and the required dependencies by running the following command:")
#     st.code("pip install streamlit-authenticator")
#     st.markdown("After installing the dependencies, navigate to the directory containing the app file and run the following command:")
#     st.code("streamlit run app.py")

#     # Contact Information
#     st.title("Contact Information")
#     st.write("These are the links to the project's repository on GitHub and articles on Medium and LinkedIn:")
#     st.markdown("- [GitHub](https://github.com/Ibrahim-I-Babana/Customer_Churn_Prediction_App.git) 🌍")
#     st.markdown("- [LinkedIn](https://www.linkedin.com/in/babana-issahak-ibrahim/) 🌐")
#     st.markdown("- [Medium](https://medium.com/@lostangelib80) ✍")

# if __name__ == "__main__":
#     main()
