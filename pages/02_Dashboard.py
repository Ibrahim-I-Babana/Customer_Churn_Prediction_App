# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Function to load the dataset
# def load_data(file_path):
#     """Function to load the dataset."""
#     df = pd.read_csv(file_path)
#     return df

# # Function to display the dashboard page
# def dashboard_page(df_concat):
#     """Function to display the dashboard page."""
#     st.title('Telecommunications Customer Churn Prediction Dashboard')
    
#     # Display dashboard image
#     st.image("Images/Customer_Churn_Prediction_Dasboard.png", caption="Telecommunications Customer Churn Prediction Dashboard", use_column_width=True)

#     # Display sample data
#     st.subheader('Sample Data')
#     st.write(df_concat.head())

#     # Display summary statistics
#     st.subheader('Summary Statistics')
#     st.write(df_concat.describe())

#     # Display distribution of target variable (Churn)
#     st.subheader('Distribution of Target Variable (Churn)')
#     fig, ax = plt.subplots()
#     sns.countplot(x='Churn', data=df_concat, palette='viridis', ax=ax)
#     st.pyplot(fig)

#     # Additional Visualizations and Insights
#     st.subheader('Additional Insights')

#     # Plot pie chart for Internet Service category
#     for service_type in df_concat['InternetService'].unique():
#         fig, ax = plt.subplots()
#         churn_counts = df_concat[df_concat['InternetService'] == service_type]['Churn'].value_counts()
#         churn_counts.plot.pie(labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'], ax=ax)
#         ax.set_title(f'Churn Distribution for {service_type}')
#         st.pyplot(fig)

#     # Plot Contract Type distribution
#     st.subheader('Contract Type vs. Churn')
#     fig, ax = plt.subplots(figsize=(15, 5))
#     sns.countplot(x='Contract', hue='Churn', data=df_concat, ax=ax)
#     st.pyplot(fig)

#     # Plotting payment method distribution
#     st.subheader('Payment Method vs. Churn')
#     fig, ax = plt.subplots()
#     sns.countplot(x='PaymentMethod', hue='Churn', data=df_concat, ax=ax)
#     ax.tick_params(axis='x', rotation=45)
#     st.pyplot(fig)

# if __name__ == "__main__":
#     # Load dataset
#     file_path = (r'C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\df_concat.csv')
#     df_concat = load_data(file_path)

#     # Display dashboard page
#     dashboard_page(df_concat)




# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pyodbc

# st.set_page_config(
#     page_title='Dashboard',
#     page_icon=':)',
#     layout='wide'
# )

# st.title('Telecommunications Data Dashboard')

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
# @st.cache
# def query_database(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         rows = cur.fetchall()
#         df = pd.DataFrame.from_records(data=rows, columns=[column[0] for column in cur.description])
        
#     return df

# # Function to plot payment method distribution.
# def plot_payment_method_distribution(df):
#     st.subheader('Payment Method Distribution')
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.countplot(x='PaymentMethod', data=df, palette='viridis', ax=ax)
#     ax.set_title('Payment Method Distribution')
#     ax.set_xlabel('Payment Method')
#     ax.set_ylabel('Count')
#     st.pyplot(fig)

# # Function to plot Contract Type distribution.
# def plot_contract_type_distribution(df):
#     st.subheader('Contract Type Distribution')
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.countplot(x='Contract', data=df, palette='viridis', ax=ax)
#     ax.set_title('Contract Type Distribution')
#     ax.set_xlabel('Contract Type')
#     ax.set_ylabel('Count')
#     st.pyplot(fig)

# # Function to plot pie chart for Internet Service category.
# def plot_internet_service_pie(df):
#     st.subheader('Internet Service Distribution')
#     internet_service_counts = df['InternetService'].value_counts()
#     fig, ax = plt.subplots()
#     ax.pie(internet_service_counts, labels=internet_service_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(internet_service_counts)))
#     ax.axis('equal')
#     ax.set_title('Internet Service Distribution')
#     st.pyplot(fig)

# # Function to display distribution of target variable (Churn).
# def display_churn_distribution(df):
#     st.subheader('Churn Distribution')
#     churn_counts = df['Churn'].value_counts()
#     st.write(churn_counts)

# # Function to display summary statistics.
# def display_summary_statistics(df):
#     st.subheader('Summary Statistics')
#     st.write(df.describe())

# # Function to display sample data.
# def display_sample_data(df, num_samples=5):
#     st.subheader('Sample Data')
#     st.write(df.sample(num_samples))

# if __name__ == "__main__":
#     # Query the database
#     data = query_database("select * from dbo.LP2_Telco_churn_first_3000")
    
#     # Display requested visualizations and calculations
#     plot_payment_method_distribution(data)
#     plot_contract_type_distribution(data)
#     plot_internet_service_pie(data)
#     display_churn_distribution(data)
#     display_summary_statistics(data)
#     display_sample_data(data)

#     st.write(st.session_state)



# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import plotly.express as px
# import altair as alt
# import matplotlib.pyplot as plt
# import numpy as np
# import pyodbc

# st.set_page_config(
#     page_title='Dashboard',
#     page_icon=':)',
#     layout='wide'
# )

# st.title('Telecommunications Customer Churn Dashboard')

# #Display dashboard image
# st.image("Images/Customer_Churn_Prediction_Dasboard.png", caption="Telecommunications Customer Churn Prediction Dashboard", use_column_width=True)

# # Initialize connection.
# @st.cache_resource
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


# # Function to plot payment method distribution.
# def plot_payment_method_distribution(df):
#     st.subheader('Payment Method Distribution')
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.countplot(x='PaymentMethod', data=df, palette='viridis', ax=ax)
#     ax.set_title('Payment Method Distribution')
#     ax.set_xlabel('Payment Method')
#     ax.set_ylabel('Count')
#     st.pyplot(fig)

# # Function to plot Contract Type distribution.
# def plot_contract_type_distribution(df):
#     st.subheader('Contract Type Distribution')
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.countplot(x='Contract', data=df, palette='viridis', ax=ax)
#     ax.set_title('Contract Type Distribution')
#     ax.set_xlabel('Contract Type')
#     ax.set_ylabel('Count')
#     st.pyplot(fig)

# # Function to plot pie chart for Internet Service category.
# def plot_internet_service_pie(df):
#     st.subheader('Internet Service Distribution')
#     internet_service_counts = df['InternetService'].value_counts()
#     fig, ax = plt.subplots()
#     ax.pie(internet_service_counts, labels=internet_service_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(internet_service_counts)))
#     ax.axis('equal')
#     ax.set_title('Internet Service Distribution')
#     st.pyplot(fig)

# # Function to display distribution of target variable (Churn).
# def display_churn_distribution(df):
#     st.subheader('Churn Distribution')
#     churn_counts = df['Churn'].value_counts()
#     st.write(churn_counts)

# # Function to display summary statistics.
# def display_summary_statistics(df):
#     st.subheader('Summary Statistics')
#     st.write(df.describe())

# # Function to display sample data.
# def display_sample_data(df, num_samples=5):
#     st.subheader('Sample Data')
#     st.write(df.sample(num_samples))

# if __name__ == "__main__":
#     # Query the database
#     data = query_database("select * from dbo.LP2_Telco_churn_first_3000")
    
#     # Display requested visualizations and calculations
#     plot_payment_method_distribution(data)
#     plot_contract_type_distribution(data)
#     plot_internet_service_pie(data)
#     display_churn_distribution(data)
#     display_summary_statistics(data)
#     display_sample_data(data)

#     st.write(st.session_state)



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from streamlit_metrics import metric, metric_row
import pygal
import leather
import plotly.express as px
 
# Load the dataset
dataset_path = (r'C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\df_concat.csv')
df = pd.read_csv(dataset_path)
 
# Convert 'TotalCharges' column to numerical values
#df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# st.title('Telecommunications Customer Churn Dashboard')

#Display dashboard image
# st.image("Images/Customer_Churn_Prediction_Dasboard.png", caption="Telecommunications Customer Churn Prediction Dashboard", use_column_width=True)
st.title('Telecommunications Customer Churn Dashboard')

# #Display dashboard image
# st.image("Images/Customer_Churn_Prediction_Dasboard.png", caption="Telecommunications Customer Churn Prediction Dashboard", use_column_width=True)

 
# # Set page title
# st.set_page_config(page_title="Visualization Dashboard")
 
# # Title for the page
# st.title("Visualization Dashboard")
 
# Sidebar navigation
option = st.sidebar.selectbox(
    'Select:',
    ('Analytics Dashboard', 'Key Performance Indicators for Churn Prediction')
)
 
if option == 'Analytics Dashboard':
    # Research question 1: Distribution of churn for different Internet service types
    st.write("Research question 1: Distribution of churn for different Internet service types")
 
    # Using Plotly Express
    fig = px.bar(df, x='InternetService', color='Churn', barmode='group',
                title='Churn Distribution for Internet Service Types (Plotly Express)',
                category_orders={'InternetService': ['DSL', 'Fiber optic', 'No']},
                color_discrete_map={'No': 'lightgreen', 'Yes': 'yellow'})
    fig.update_xaxes(title="Internet Service Type")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig)
 
    # Research question 2: Impact of having a partner or dependents on customer churn
    st.write("Research question 2: Impact of having a partner or dependents on customer churn")
 
    # Using Altair
    partner_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Partner:O', title='Partner Status'),
        y=alt.Y('count():Q', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution for Partner Status (Altair)"
    )
    st.altair_chart(partner_chart, use_container_width=True)
 
    dependents_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Dependents:O', title='Dependents Status'),
        y=alt.Y('count():Q', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution for Dependents Status (Altair)"
    )
    st.altair_chart(dependents_chart, use_container_width=True)
 
    # Research question 3: Influence of contract type on customer churn
    st.header("Research question 3: Influence of contract type on customer churn")
 
    # Using Plotly Express
    fig2 = px.histogram(df, x='Contract', color='Churn', barmode='group')
    fig2.update_layout(title="Churn Distribution for Contract Type (Plotly Express)")
    st.plotly_chart(fig2, use_container_width=True)
 
    # Research question 4: Impact of billing preference on customer churn
    st.header("Research question 4: Impact of billing preference on customer churn")
 
    # Convert 'Churn' column to boolean (0 for No, 1 for Yes)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
 
    # Group data by Billing Preference and calculate churn
    billing_churn = df.groupby('PaperlessBilling')['Churn'].sum().reset_index()
 
    # Plot using Plotly Express
    fig = px.bar(billing_churn, x='PaperlessBilling', y='Churn',
                labels={'PaperlessBilling': 'Billing Preference', 'Churn': 'Churn Count'},
                title='Churn Distribution for Billing Preference (Plotly Express)')
    st.plotly_chart(fig)
 
 
 
    # Using Altair
    gender_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('gender', title='Gender'),
        y=alt.Y('count()', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution by Gender (Altair)"
    )
    st.altair_chart(gender_chart, use_container_width=True)
 
    # Additional research questions
    st.header("Additional Research Questions")
 
    # Research question 6: Impact of tenure on customer churn
    st.header("Research question 6: What is the impact of tenure on customer churn")
 
    # Plot using Plotly Express
    fig = px.histogram(df, x='tenure', color='Churn', nbins=20,
                    labels={'tenure': 'Tenure', 'Churn': 'Churn'},
                    title='Impact of Tenure on Customer Churn')
    st.plotly_chart(fig)
 
    # Research question 7: Relationship between total charges and churn
    st.subheader("Research question 7: Relationship between total charges and churn")
    charges_churn_scatter = alt.Chart(df).mark_circle(size=60).encode(
        x='TotalCharges',
        y='Churn',
        color='Churn:N',
        tooltip=['TotalCharges', 'Churn']
    ).properties(
        title="Churn vs Total Charges (Altair)"
    ).interactive()
    st.altair_chart(charges_churn_scatter, use_container_width=True)
 
 
   
elif option == 'Key Performance Indicators for Churn Prediction':
    # Key Performance Indicators (KPIs)
    st.header("Key Performance Indicators (KPIs)")
 
    # Calculate Gross MRR Churn
    gross_mrr_churn = 0.05  # Example value, replace with actual calculation
 
    # Calculate Net MRR Churn
    net_mrr_churn = 0.03  # Example value, replace with actual calculation
 
    # Calculate Net Change in Customers
    net_change_customers = 100  # Example value, replace with actual calculation
 
    # Calculate Revenue Growth Rate
    revenue_growth_rate = 0.10  # Example value, replace with actual calculation
 
    # Calculate Activation Rate
    activation_rate = 0.75  # Example value, replace with actual calculation
 
    # Calculate DAU/MAU Ratio
    dau_mau_ratio = 0.65  # Example value, replace with actual calculation
 
    # Calculate Net Promoter Score (NPS)
    nps = 75  # Example value, replace with actual calculation
 
    # Calculate Customer Satisfaction Score (CSAT)
    csat = 85  # Example value, replace with actual calculation
 
    # Calculate Customer Lifetime Value (LTV)
    clv = 1500  # Example value, replace with actual calculation
 
    # Display metrics in columns
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("Financial Metrics")
        metric("Gross MRR Churn", f"<span style='color:green'>{gross_mrr_churn}</span>")
        metric("Net MRR Churn", f"<span style='color:yellow'>{net_mrr_churn}</span>")
        metric("Net Change in Customers", f"<span style='color:blue'>{net_change_customers}</span>")
        metric("Revenue Growth Rate", f"<span style='color:pink'>{revenue_growth_rate}</span>")
 
    with col2:
        st.subheader("Product Metrics")
        metric("Activation Rate", f"<span style='color:purple'>{activation_rate}</span>")
        metric("DAU/MAU Ratio", f"<span style='color:orange'>{dau_mau_ratio}</span>")
        metric("Net Promoter Score (NPS)", f"<span style='color:green'>{nps}</span>")
        metric("Customer Satisfaction (CSAT)", f"<span style='color:yellow'>{csat}</span>")
        metric("Customer Lifetime Value (LTV)", f"<span style='color:blue'>{clv}</span>")