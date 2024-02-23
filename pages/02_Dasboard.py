# import streamlit as st


# st.set_page_config(page_title='Dashboard'
#                    page_icon=':)'initial_sidebar_state='auto'
#                    page_layout='wide'
#                    )


# st.title('Customer Churn Prediction Dashboard')

# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
 
# def dashboard_page(df_concat, dashboard_image_path):
#     st.title('Telecommunications Customer Churn Prediction Dashboard')
#     st.image('C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\Images\Global Telecommunications Network.jpg, use_column_width=True')
 
#     # Display sample data
#     st.write('Sample Data:')
#     st.write(df_concat.head())
 
#     # Display summary statistics
#     st.subheader('Summary Statistics')
#     st.write(df_concat.describe())
 
#     # Display distribution of target variable
#     st.subheader('Distribution of Target Variable (Churn)')
#     sns.countplot(x='Churn', data=df_concat, palette='viridis')
#     st.pyplot()

#     # Plot a pie chart for each internet service category
# plt.figure(figsize=(12, 5))

# for i, service_type in enumerate(churn_counts.index):
#     plt.subplot(1, 3, i + 1)
#     plt.pie(churn_counts.loc[service_type], labels=churn_counts.columns, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
#     plt.title(f'Churn Distribution for {service_type}')

# plt.tight_layout()
# plt.show()

# # Set the style of seaborn for better visualization
# sns.set(style="whitegrid")

# # Plotting contract type distribution
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# sns.countplot(x='Contract', hue='Churn', data=df_concat)
# plt.title('Contract Type vs. Churn')

# # Adjust layout
# plt.tight_layout()
# plt.show()
 
#     # Add more visualizations and insights as needed
 
# if __name__ == "__main__":
#     # Load your dataset here
#     df = pd.read_csv('C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\df_concat.csv')
#     dashboard_image_path = ('C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\Images\Customer_Churn_Prediction_Dasboard.png')
#     dashboard_page(df_concat, dashboard_image_path)




# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# def load_data('C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Customer_Churn_Prediction_App/df_concat.csv')
#     """Function to load the dataset."""
#     df = pd.read_csv(file_path)
#     return df

# def dashboard_page(df_concat):
#     """Function to display the dashboard page."""
#     st.title('Telecommunications Customer Churn Prediction Dashboard')
    
#     # Display dashboard image
#     st.image('C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\Images\Customer_Churn_Prediction_Dasboard.png', use_column_width=True)

#     # Display sample data
#     st.subheader('Sample Data')
#     st.write(df_concat.head())

#     # Display summary statistics
#     st.subheader('Summary Statistics')
#     st.write(df_concat.describe())

#     # Display distribution of target variable (Churn)
#     st.subheader('Distribution of Target Variable (Churn)')
#     sns.countplot(x='Churn', data=df_concat, palette='viridis')
#     st.pyplot()

#     # Additional Visualizations and Insights
#     st.subheader('Additional Insights')

#     # Plot pie chart for Internet Service category
#     churn_counts = df_concat.groupby('InternetService')['Churn'].value_counts().unstack()
#     plt.figure(figsize=(12, 5))
#     for service_type in churn_counts.index:
#         plt.pie(churn_counts.loc[service_type], labels=churn_counts.columns, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
#         plt.title(f'Churn Distribution for {service_type}')
#         st.pyplot()

#     # Plot Contract Type distribution
#     plt.figure(figsize=(15, 5))
#     sns.countplot(x='Contract', hue='Churn', data=df_concat)
#     plt.title('Contract Type vs. Churn')
#     st.pyplot()

#     # Add more visualizations and insights as needed

# if __name__ == "__main__":
#     # Load dataset
#     file_path = ('C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\df_concat.csv')
#     df = load_data(file_path)

#     # Display dashboard page
#     dashboard_page(df_concat)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """Function to load the dataset."""
    df = pd.read_csv(file_path)
    return df

def dashboard_page(df_concat):
    """Function to display the dashboard page."""
    st.title('Telecommunications Customer Churn Prediction Dashboard')
    
    # Display dashboard image
    st.image(r'C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\Images\Customer_Churn_Prediction_Dasboard.png', use_column_width=True)

    # Display sample data
    st.subheader('Sample Data')
    st.write(df_concat.head())

    # Display summary statistics
    st.subheader('Summary Statistics')
    st.write(df_concat.describe())

    # Display distribution of target variable (Churn)
    st.subheader('Distribution of Target Variable (Churn)')
    sns.countplot(x='Churn', data=df_concat, palette='viridis')
    st.pyplot()

    # Additional Visualizations and Insights
    st.subheader('Additional Insights')

    # Plot pie chart for Internet Service category
    churn_counts = df_concat.groupby('InternetService')['Churn'].value_counts().unstack()
    plt.figure(figsize=(12, 5))
    for service_type in churn_counts.index:
        plt.pie(churn_counts.loc[service_type], labels=churn_counts.columns, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
        plt.title(f'Churn Distribution for {service_type}')
        st.pyplot()

    # Plot Contract Type distribution
    plt.figure(figsize=(15, 5))
    sns.countplot(x='Contract', hue='Churn', data=df_concat)
    plt.title('Contract Type vs. Churn')
    st.pyplot()

    # Add more visualizations and insights as needed

if __name__ == "__main__":
    # Load dataset
    file_path = (r'C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\df_concat.csv')
    df_concat = load_data(file_path)

    # Display dashboard page
    dashboard_page(df_concat)

