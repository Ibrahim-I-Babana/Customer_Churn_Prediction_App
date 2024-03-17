Customer Churn Prediction App
Losing customers is a significant concern for any organization. Customer churn, also known as customer attrition or turnover, refers to the percentage of customers who stop using a company's product or service within a specific period. For instance, if a company starts the year with 500 customers and ends with 480, the churn rate would be 4%. Understanding the reasons and timing of customer churn can greatly assist organizations in devising effective retention strategies.

Introduction
This data application focuses on predicting customer churn at a telecommunications company. The project has successfully developed robust machine learning models, and now the aim is to deploy these models for practical decision-making. Effective deployment is crucial for unlocking actionable insights and realizing the full impact of the models. Following the CRISP-DM framework, model deployment seamlessly integrates machine learning models into existing production environments, facilitating informed business decisions based on data.

Project Focus
The main focus of the project is to create a user-friendly Graphic User Interface (GUI) using Streamlit. This GUI will enable stakeholders without technical expertise to interact with the machine learning models effectively.

Project Benefits
Customizable GUI Components: Learn to craft flexible and user-friendly GUI components with Streamlit.
Real-time Predictions: Deploy machine learning models using Streamlit for real-time predictions in practical scenarios.
CRISP-DM Final Step: Embrace the industry-standard CRISP-DM framework to guide the data process from business understanding to model deployment.
Data
The dataset consists of 5000 records with attributes ranging from demographic information to service usage and churn status, hosted in a Microsoft SQL server.

Project Structure
Customer_Churn_Prediction_App
├── data
│
├── notebook: P4 Customer Churn Prediction
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Preprocessing_and_Feature_Engineering.ipynb
│   ├── 03_Model_Application.ipynb
│   └── ...
├── visuals
│   ├── EDA-Correlation Matrix.png
│   ├── Hyperparameter tuning results.png
│   ├── Churn Distribution by Internet Service Types.png
│   └── Churn Distribution by Contract Types.png
├── README.md
└── requirements.txt
Notebook
Explore detailed analyses in the notebook:

01_Exploratory_Data_Analysis: In-depth exploration of sales data.
02_Preprocessing_and_Feature_Engineering: Preprocessing steps and feature engineering.
03_Model_Application: Application of One-Hot Encoding, Linear Regression, Hyperparameter Tuning, and Random Forests.
Visuals
Discover visual representations in the visuals folder, aiding in understanding complex patterns and trends.

Requirements
The necessary Python packages are outlined in 'requirements.txt'.