# import streamlit as st
# import pandas as pd
# import os
# import datetime
# import joblib
# from sklearn.preprocessing import OneHotEncoder

# st.set_page_config(
#     page_title='Dashboard',
#     page_icon=':)',
#     layout='wide'
# )

# @st.cache_resource(show_spinner='Model Loading')
# def load_best_rf_model():
#     pipeline = joblib.load('../Customer_Churn_Prediction_App/models/best_rf_model.joblib')
#     return pipeline

# @st.cache_resource(show_spinner='Model Loading')
# def load_best_dt_model():
#     pipeline = joblib.load('../Customer_Churn_Prediction_App/models/best_dt_model.joblib')
#     return pipeline

# def select_model():
#     col1, col2 = st.columns(2)

#     with col1:
#         st.session_state['selected_model'] = st.selectbox('Select a Model', options=[
#                      'RandomForestClassifier', 'DecisionTreeClassifier'], key='select_model')
    
#     with col2:
#         pass
    
#     if st.session_state['selected_model'] == 'RandomForestClassifier':
#         pipeline = load_best_rf_model()
#     else:
#         pipeline = load_best_dt_model()
    
#     encoder = joblib.load('./models/pipeline.joblib')

#     return pipeline, encoder


# if 'prediction' not in st.session_state:
#     st.session_state['prediction'] = None
# if 'probability' not in st.session_state:
#     st.session_state['probability'] = None

# def preprocess_input_data(df):
#     categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
#                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
#     # Apply one-hot encoding to categorical columns
#     encoder = OneHotEncoder(drop='first')
#     encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
#     # Replace original categorical columns with the encoded columns
#     df = pd.concat([df.drop(columns=categorical_cols), encoded_cols], axis=1)


#     # #if not os.path.exists("./data/history.csv"):
#     #       #os.makedirs("./data/history.csv")
#     # df.to_csv("./data/history.csv",mode='a',header=not os.path.exists('./data/history.csv'),index=False)

#     return df, encoder


# def make_prediction(pipeline, encoder):
#     # Get input features
#     gender = st.session_state['gender']
#     seniorcitizen = st.session_state['seniorcitizen']
#     partner = st.session_state['partner']
#     dependents = st.session_state['dependents']
#     tenure = st.session_state['tenure']
#     phoneservice = st.session_state['phoneservice']
#     multiplelines = st.session_state['multiplelines']
#     internetservice = st.session_state['internetservice']
#     onlinesecurity = st.session_state['onlinesecurity']
#     onlinebackup = st.session_state['onlinebackup']
#     deviceprotection = st.session_state['deviceprotection']
#     techsupport = st.session_state['techsupport']
#     streamingtv = st.session_state['streamingtv']
#     streamingmovies = st.session_state['streamingmovies']
#     contract = st.session_state['contract']
#     paperlessbilling = st.session_state['paperlessbilling']
#     paymentmethod = st.session_state['paymentmethod']
#     monthlycharges = st.session_state['monthlycharges']
#     totalcharges = st.session_state['totalcharges']



#     # Create input data DataFrame
#     input_data = ({
#         'gender': [gender],
#         'SeniorCitizen': [seniorcitizen],
#         'Partner': [partner],
#         'Dependents': [dependents],
#         'tenure': [tenure],
#         'PhoneService': [phoneservice],
#         'MultipleLines': [multiplelines],
#         'InternetService': [internetservice],
#         'OnlineSecurity': [onlinesecurity],
#         'OnlineBackup': [onlinebackup],
#         'DeviceProtection': [deviceprotection],
#         'TechSupport': [techsupport],
#         'StreamingTV': [streamingtv],
#         'StreamingMovies': [streamingmovies],
#         'Contract': [contract],
#         'PaperlessBilling': [paperlessbilling],
#         'PaymentMethod': [paymentmethod],
#         'MonthlyCharges': [monthlycharges],
#         'TotalCharges': [totalcharges]
#     })

#     # Preprocess input data
#     input_data_encoded = encoder.transform(input_data)

#     # Make prediction
#     prediction = pipeline.predict(input_data_encoded)
#     probability = pipeline.predict_proba(input_data_encoded)

#     # Update session state
#     st.session_state['prediction'] = prediction
#     st.session_state['probability'] = probability

#     return prediction, probability


# def display_form():
#     pipeline, encoder = select_model()

#     with st.form('input-feature'):
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.markdown('### **Demographics** ###')
#             st.session_state['gender'] = st.radio('Gender', ["Male", "Female"])
#             st.session_state['seniorcitizen'] = st.radio('Senior Citizen', ["No", "Yes"])
#             st.session_state['partner'] = st.radio('Partner', ["Yes", "No"])
#             st.session_state['dependents'] = st.radio('Dependents', ["Yes", "No"])
#             st.session_state['tenure'] = st.number_input('Length of Tenure with the Telco (in months)', min_value=1, max_value=71, step=1)

#         with col2:
#             st.markdown('### **Charges** ###')
#             st.session_state['monthlycharges'] = st.number_input('Monthly Charges', value=0.0)
#             st.session_state['totalcharges'] = st.number_input('Total Charges', value=0.0)

#         with col3:
#             st.markdown('### **Billing Method** ###')
#             st.session_state['contract'] = st.selectbox('Contract', ["Month-to-month", "One year", "Two year"])
#             st.session_state['paperlessbilling'] = st.radio('Paperless Billing', ["Yes", "No"])
#             st.session_state['paymentmethod'] = st.selectbox('Payment Method', ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
#         with col4:
#             st.markdown('### **Services** ###')
#             st.session_state['phoneservice'] = st.radio('Phone Service', ["Yes", "No"])
#             st.session_state['multiplelines'] = st.selectbox('Multiple Lines', ["Yes", "No", "No phone service"])
#             st.session_state['internetservice'] = st.selectbox('Internet Service', ["DSL", "Fiber optic", "No"])
#             st.session_state['onlinesecurity'] = st.selectbox('Online Security', ["Yes", "No", "No internet service"])
#             st.session_state['onlinebackup'] = st.selectbox('Online Backup', ["Yes", "No", "No internet service"])
#             st.session_state['deviceprotection'] = st.selectbox('Device Protection', ["Yes", "No", "No internet service"])
#             st.session_state['techsupport'] = st.selectbox('Tech Support', ["Yes", "No", "No internet service"])
#             st.session_state['streamingtv'] = st.selectbox('Streaming TV', ["Yes", "No", "No internet service"])
#             st.session_state['streamingmovies'] = st.selectbox('Streaming Movies', ["Yes", "No", "No internet service"])
        
#         st.form_submit_button('Submit', on_click=make_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder))

# def main():
#     st.set_page_config(page_title="Churn Prediction", layout="wide")
#     st.title("Customer Churn Predictor")




# if __name__ == "__main__":
#     st.title('Customer Churn Predictor')
#     display_form()

#     prediction = st.session_state['prediction']
#     probability = st.session_state['probability']

#     if prediction is None:
#         st.markdown('### Predictions will show here')
#     elif prediction == "Yes":
#         probability_of_yes = probability[0][1] * 100
#         st.markdown(f"### The Customer will Churn with a probability of {round(probability_of_yes, 2)}%")
                    
#     else:
#         probability_of_no = probability[0][0] * 100
#         st.markdown(f"### The Customer will not Churn with a probability of {round(probability_of_no, 2)}% ")

#     st.write(st.session_state)








# import streamlit as st
# import pandas as pd
# import os
# import datetime
# import joblib
# from sklearn.preprocessing import OneHotEncoder

# # Load models
# @st.cache_resource(show_spinner='Model Loading')
# def load_best_rf_model():
#     pipeline = joblib.load('../Customer_Churn_Prediction_App/models/best_rf_model.joblib')
#     return pipeline

# @st.cache_resource(show_spinner='Model Loading')
# def load_best_dt_model():
#     pipeline = joblib.load('../Customer_Churn_Prediction_App/models/best_dt_model.joblib')
#     return pipeline

# # Select model
# def select_model():
#     col1, col2 = st.columns(2)

#     with col1:
#         st.session_state['selected_model'] = st.selectbox('Select a Model', options=[
#                      'RandomForestClassifier', 'DecisionTreeClassifier'], key='select_model')
    
#     with col2:
#         pass
    
#     if st.session_state['selected_model'] == 'RandomForestClassifier':
#         pipeline = load_best_rf_model()
#     else:
#         pipeline = load_best_dt_model()
    
#     encoder = joblib.load('./models/pipeline.joblib')

#     return pipeline, encoder

# # Preprocess input data
# def preprocess_input_data(df):
#     categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
#                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
#     # Apply one-hot encoding to categorical columns
#     encoder = OneHotEncoder(drop='first')
#     encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
#     # Replace original categorical columns with the encoded columns
#     df = pd.concat([df.drop(columns=categorical_cols), encoded_cols], axis=1)
    
#     df['Prediction Time'] = datetime.date.today()
#     df['Model Used'] = st.session_state['selected_model']
    

#     # Save data to CSV file
#     if not os.path.exists("./data/history.csv"):
#         os.makedirs("./data/history.csv")
#     df.to_csv("./data/history.csv", mode='a', header=not os.path.exists('./data/history.csv'), index=False)

#     return df, encoder

# # Make prediction
# def make_prediction(pipeline, encoder):
#     # Get input features
#     gender = st.session_state['gender']
#     seniorcitizen = st.session_state['seniorcitizen']
#     partner = st.session_state['partner']
#     dependents = st.session_state['dependents']
#     tenure = st.session_state['tenure']
#     phoneservice = st.session_state['phoneservice']
#     multiplelines = st.session_state['multiplelines']
#     internetservice = st.session_state['internetservice']
#     onlinesecurity = st.session_state['onlinesecurity']
#     onlinebackup = st.session_state['onlinebackup']
#     deviceprotection = st.session_state['deviceprotection']
#     techsupport = st.session_state['techsupport']
#     streamingtv = st.session_state['streamingtv']
#     streamingmovies = st.session_state['streamingmovies']
#     contract = st.session_state['contract']
#     paperlessbilling = st.session_state['paperlessbilling']
#     paymentmethod = st.session_state['paymentmethod']
#     monthlycharges = st.session_state['monthlycharges']
#     totalcharges = st.session_state['totalcharges']

#     # Create input data DataFrame
#     input_data = pd.DataFrame({
#         'gender': [gender],
#         'SeniorCitizen': [seniorcitizen],
#         'Partner': [partner],
#         'Dependents': [dependents],
#         'tenure': [tenure],
#         'PhoneService': [phoneservice],
#         'MultipleLines': [multiplelines],
#         'InternetService': [internetservice],
#         'OnlineSecurity': [onlinesecurity],
#         'OnlineBackup': [onlinebackup],
#         'DeviceProtection': [deviceprotection],
#         'TechSupport': [techsupport],
#         'StreamingTV': [streamingtv],
#         'StreamingMovies': [streamingmovies],
#         'Contract': [contract],
#         'PaperlessBilling': [paperlessbilling],
#         'PaymentMethod': [paymentmethod],
#         'MonthlyCharges': [monthlycharges],
#         'TotalCharges': [totalcharges]
#     })

#     # Preprocess input data
#     input_data_encoded = encoder.transform(input_data)

#     # Make prediction
#     prediction = pipeline.predict(input_data_encoded)
#     probability = pipeline.predict_proba(input_data_encoded)

#     # Update session state
#     st.session_state['prediction'] = prediction
#     st.session_state['probability'] = probability

#     return prediction, probability

# # Display form for input
# def display_form():
#     pipeline, encoder = select_model()

#     with st.form('input-feature'):
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.markdown('### **Demographics** ###')
#             st.session_state['gender'] = st.radio('Gender', ["Male", "Female"])
#             st.session_state['seniorcitizen'] = st.radio('Senior Citizen', ["No", "Yes"])
#             st.session_state['partner'] = st.radio('Partner', ["Yes", "No"])
#             st.session_state['dependents'] = st.radio('Dependents', ["Yes", "No"])
#             st.session_state['tenure'] = st.number_input('Length of Tenure with the Telco (in months)', min_value=1, max_value=71, step=1)

#         with col2:
#             st.markdown('### **Charges** ###')
#             st.session_state['monthlycharges'] = st.number_input('Monthly Charges', value=0.0)
#             st.session_state['totalcharges'] = st.number_input('Total Charges', value=0.0)

#         with col3:
#             st.markdown('### **Billing Method** ###')
#             st.session_state['contract'] = st.selectbox('Contract', ["Month-to-month", "One year", "Two year"])
#             st.session_state['paperlessbilling'] = st.radio('Paperless Billing', ["Yes", "No"])
#             st.session_state['paymentmethod'] = st.selectbox('Payment Method', ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
#         with col4:
#             st.markdown('### **Services** ###')
#             st.session_state['phoneservice'] = st.radio('Phone Service', ["Yes", "No"])
#             st.session_state['multiplelines'] = st.selectbox('Multiple Lines', ["Yes", "No", "No phone service"])
#             st.session_state['internetservice'] = st.selectbox('Internet Service', ["DSL", "Fiber optic", "No"])
#             st.session_state['onlinesecurity'] = st.selectbox('Online Security', ["Yes", "No", "No internet service"])
#             st.session_state['onlinebackup'] = st.selectbox('Online Backup', ["Yes", "No", "No internet service"])
#             st.session_state['deviceprotection'] = st.selectbox('Device Protection', ["Yes", "No", "No internet service"])
#             st.session_state['techsupport'] = st.selectbox('Tech Support', ["Yes", "No", "No internet service"])
#             st.session_state['streamingtv'] = st.selectbox('Streaming TV', ["Yes", "No", "No internet service"])
#             st.session_state['streamingmovies'] = st.selectbox('Streaming Movies', ["Yes", "No", "No internet service"])
        
#         st.form_submit_button('Make your Prediction', on_click=make_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder))

# # Batch prediction logic
# def batch_prediction():
#     upload_file = st.file_uploader("Upload CSV", type=["csv"])
#     if upload_file is not None:
#         df = pd.read_csv(upload_file)
#         script_directory = os.path.dirname(os.path.abspath(__file__))
#         # Update the path to the preprocessor model file
#         preprocessor_path = os.path.join(script_directory, "..", "batch_models", "batch_pipeline_preprocessor.pkl")
#         with open(preprocessor_path, 'rb') as file:
#             preprocessor = joblib.load(file)
#         X, y = preprocessor(df)
#         model_option = st.selectbox("Select Model", ["SVC", "XGBoost"])
#         if st.button("Predict"):
#             if model_option == "SVC":
#                 svc_model_path = os.path.join(script_directory, "..", "batch_models", "svc_model.pkl")
#                 svc_model = joblib.load(svc_model_path)
#                 churn_percentage = predict(svc_model, X)  # Implement predict() function
#             else:
#                 xgb_model_path = os.path.join(script_directory, "..", "batch_models", "xgb_model.pkl")
#                 xgb_model = joblib.load(xgb_model_path)
#                 churn_percentage = predict(xgb_model, X)  # Implement predict() function
#             st.success(f"Churn Percentage ({model_option} Model): {churn_percentage:.2f}%")

# # Main function
# def main():
#     st.set_page_config(page_title='Churn Prediction', layout='wide')
#     st.title('Customer Churn Predictor')
    
#     prediction_option = st.radio("Select Prediction Option", ["Online", "Batch"])

#     if prediction_option == "Online":
#         display_form()
#     elif prediction_option == "Batch":
#         batch_prediction()
    
#     prediction = st.session_state['prediction']
#     probability = st.session_state['probability']

#     if prediction is None:
#         st.markdown('### Predictions will show here')
#     elif prediction == "Yes":
#         probability_of_yes = probability[0][1] * 100
#         st.markdown(f"### The Customer will Churn with a probability of {round(probability_of_yes, 2)}%")
                    
#     else:
#         probability_of_no = probability[0][0] * 100
#         st.markdown(f"### The Customer will not Churn with a probability of {round(probability_of_no, 2)}% ")



#     # Display session state
#     st.write(st.session_state)

# if __name__ == "__main__":
#     main()


# if __name__ == "__main__":
#     st.title('Customer Churn Predictor')
#     display_form()

#     prediction = st.session_state['prediction']
#     probability = st.session_state['probability']

#     if prediction is None:
#         st.markdown('### Predictions will show here')
#     elif prediction == "Yes":
#         probability_of_yes = probability[0][1] * 100
#         st.markdown(f"### The Customer will Churn with a probability of {round(probability_of_yes, 2)}%")
                    
#     else:
#         probability_of_no = probability[0][0] * 100
#         st.markdown(f"### The Customer will not Churn with a probability of {round(probability_of_no, 2)}% ")

# #     st.write(st.session_state)





# import streamlit as st
# import pandas as pd
# import os
# import datetime
# import joblib
# from sklearn.preprocessing import OneHotEncoder

# # Load models
# @st.cache_resource(show_spinner='Model Loading')
# def load_best_rf_model():
#     pipeline = joblib.load('../Customer_Churn_Prediction_App/models/best_rf_model.joblib')
#     return pipeline

# @st.cache_resource(show_spinner='Model Loading')
# def load_best_dt_model():
#     pipeline = joblib.load('../Customer_Churn_Prediction_App/models/best_dt_model.joblib')
#     return pipeline

# # Select model
# def select_model():
#     col1, col2 = st.columns(2)

#     with col1:
#         st.session_state['selected_model'] = st.selectbox('Select a Model', options=[
#                      'RandomForestClassifier', 'DecisionTreeClassifier'], key='select_model')
    
#     with col2:
#         pass
    
#     if st.session_state['selected_model'] == 'RandomForestClassifier':
#         pipeline = load_best_rf_model()
#     else:
#         pipeline = load_best_dt_model()
    
#     encoder = joblib.load('./models/pipeline.joblib')

#     return pipeline, encoder

# # Preprocess input data
# def preprocess_input_data(df):
#     categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
#                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
#     # Apply one-hot encoding to categorical columns
#     encoder = OneHotEncoder(drop='first')
#     encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
#     # Replace original categorical columns with the encoded columns
#     df = pd.concat([df.drop(columns=categorical_cols), encoded_cols], axis=1)
    
#     # df['Prediction Time'] = datetime.date.today()
#     # df['Model Used'] = st.session_state['selected_model']
    

#     # # Save data to CSV file
#     # if not os.path.exists("./data/history.csv"):
#     #     os.makedirs("./data/History.csv")
#     df.to_csv("./History.csv", mode='a') #, header=not os.path.exists('./data/history.csv'), index=False)
    
#     # df['Prediction Time'] = datetime.date.today()
#     # df['Model Used'] = st.session_state['selected_model']


#     return df, encoder

# # Make prediction
# def make_prediction(pipeline, encoder):
#     # Get input features
#     gender = st.session_state['gender']
#     seniorcitizen = st.session_state['seniorcitizen']
#     partner = st.session_state['partner']
#     dependents = st.session_state['dependents']
#     tenure = st.session_state['tenure']
#     phoneservice = st.session_state['phoneservice']
#     multiplelines = st.session_state['multiplelines']
#     internetservice = st.session_state['internetservice']
#     onlinesecurity = st.session_state['onlinesecurity']
#     onlinebackup = st.session_state['onlinebackup']
#     deviceprotection = st.session_state['deviceprotection']
#     techsupport = st.session_state['techsupport']
#     streamingtv = st.session_state['streamingtv']
#     streamingmovies = st.session_state['streamingmovies']
#     contract = st.session_state['contract']
#     paperlessbilling = st.session_state['paperlessbilling']
#     paymentmethod = st.session_state['paymentmethod']
#     monthlycharges = st.session_state['monthlycharges']
#     totalcharges = st.session_state['totalcharges']

#     # Create input data DataFrame
#     input_data = pd.DataFrame({
#         'gender': [gender],
#         'SeniorCitizen': [seniorcitizen],
#         'Partner': [partner],
#         'Dependents': [dependents],
#         'tenure': [tenure],
#         'PhoneService': [phoneservice],
#         'MultipleLines': [multiplelines],
#         'InternetService': [internetservice],
#         'OnlineSecurity': [onlinesecurity],
#         'OnlineBackup': [onlinebackup],
#         'DeviceProtection': [deviceprotection],
#         'TechSupport': [techsupport],
#         'StreamingTV': [streamingtv],
#         'StreamingMovies': [streamingmovies],
#         'Contract': [contract],
#         'PaperlessBilling': [paperlessbilling],
#         'PaymentMethod': [paymentmethod],
#         'MonthlyCharges': [monthlycharges],
#         'TotalCharges': [totalcharges]
#     })

#     # Preprocess input data
#     input_data_encoded = encoder.transform(input_data)

#     # Make prediction
#     prediction = pipeline.predict(input_data_encoded)
#     probability = pipeline.predict_proba(input_data_encoded)

#     # Update session state
#     st.session_state['prediction'] = prediction
#     st.session_state['probability'] = probability

#     return prediction, probability

# # Display form for input
# def display_form():
#     pipeline, encoder = select_model()

#     with st.form('input-feature'):
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.markdown('### **Demographics** ###')
#             st.session_state['gender'] = st.radio('Gender', ["Male", "Female"])
#             st.session_state['seniorcitizen'] = st.radio('Senior Citizen', ["No", "Yes"])
#             st.session_state['partner'] = st.radio('Partner', ["Yes", "No"])
#             st.session_state['dependents'] = st.radio('Dependents', ["Yes", "No"])
#             st.session_state['tenure'] = st.number_input('Length of Tenure with the Telco (in months)', min_value=1, max_value=71, step=1)

#         with col2:
#             st.markdown('### **Charges** ###')
#             st.session_state['monthlycharges'] = st.number_input('Monthly Charges', value=0.0)
#             st.session_state['totalcharges'] = st.number_input('Total Charges', value=0.0)

#         with col3:
#             st.markdown('### **Billing Method** ###')
#             st.session_state['contract'] = st.selectbox('Contract', ["Month-to-month", "One year", "Two year"])
#             st.session_state['paperlessbilling'] = st.radio('Paperless Billing', ["Yes", "No"])
#             st.session_state['paymentmethod'] = st.selectbox('Payment Method', ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
#         with col4:
#             st.markdown('### **Services** ###')
#             st.session_state['phoneservice'] = st.radio('Phone Service', ["Yes", "No"])
#             st.session_state['multiplelines'] = st.selectbox('Multiple Lines', ["Yes", "No", "No phone service"])
#             st.session_state['internetservice'] = st.selectbox('Internet Service', ["DSL", "Fiber optic", "No"])
#             st.session_state['onlinesecurity'] = st.selectbox('Online Security', ["Yes", "No", "No internet service"])
#             st.session_state['onlinebackup'] = st.selectbox('Online Backup', ["Yes", "No", "No internet service"])
#             st.session_state['deviceprotection'] = st.selectbox('Device Protection', ["Yes", "No", "No internet service"])
#             st.session_state['techsupport'] = st.selectbox('Tech Support', ["Yes", "No", "No internet service"])
#             st.session_state['streamingtv'] = st.selectbox('Streaming TV', ["Yes", "No", "No internet service"])
#             st.session_state['streamingmovies'] = st.selectbox('Streaming Movies', ["Yes", "No", "No internet service"])
        
#         st.form_submit_button('Make your Prediction', on_click=make_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder))

# # Batch prediction logic
# def batch_prediction():
#     upload_file = st.file_uploader("Upload CSV", type=["csv"])
#     if upload_file is not None:
#         df = pd.read_csv(upload_file)
#         script_directory = os.path.dirname(os.path.abspath(__file__))
#         # Update the path to the preprocessor model file
#         preprocessor_path = os.path.join(script_directory, "..", "batch_models", "batch_pipeline_preprocessor.pkl")
#         with open(preprocessor_path, 'rb') as file:
#             preprocessor = joblib.load(file)
#         X, y = preprocessor(df)
#         model_option = st.selectbox("Select Model", ["SVC", "XGBoost"])
#         if st.button("Predict"):
#             if model_option == "SVC":
#                 svc_model_path = os.path.join(script_directory, "..", "batch_models", "svc_model.pkl")
#                 svc_model = joblib.load(svc_model_path)
#                 churn_percentage = predict(svc_model, X)  # Implement predict() function
#             else:
#                 xgb_model_path = os.path.join(script_directory, "..", "models", "xgb_model.pkl")
#                 xgb_model = joblib.load(xgb_model_path)
#                 churn_percentage = predict(xgb_model, X)  # Implement predict() function
#             st.success(f"Churn Percentage ({model_option} Model): {churn_percentage:.2f}%")

# # Main function
# def main():
#     st.set_page_config(page_title='Churn Prediction', layout='wide')
#     st.title('Customer Churn Predictor')
    
#     prediction_option = st.radio("Select Prediction Option", ["Online", "Batch"])

#     st.session_state.setdefault('prediction', None)
#     st.session_state.setdefault('probability', None)

#     if prediction_option == "Online":
#         display_form()
#     elif prediction_option == "Batch":
#         batch_prediction()
    
#     prediction = st.session_state['prediction']
#     probability = st.session_state['probability']

#     if prediction is None:
#         st.markdown('### Predictions will show here')
#     elif prediction == "Yes":
#         probability_of_yes = probability[0][1] * 100
#         st.markdown(f"### The Customer will Churn with a probability of {round(probability_of_yes, 2)}%")
                    
#     else:
#         probability_of_no = probability[0][0] * 100
#         st.markdown(f"### The Customer will not Churn with a probability of {round(probability_of_no, 2)}% ")


#     # Display session state
#     st.write(st.session_state)

# if __name__ == "__main__":
#     main()





import streamlit as st
import pandas as pd
import os
import datetime
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load models
@st.cache_resource #(show_spinner=False)
def load_best_rf_model():
    pipeline = joblib.load('../Customer_Churn_Prediction_App/models/best_rf_model.joblib')
    return pipeline

@st.cache_resource #(show_spinner=False)
def load_best_dt_model():
    pipeline = joblib.load('../Customer_Churn_Prediction_App/models/best_dt_model.joblib')
    return pipeline

# Select model
def select_model():
    col1, col2 = st.columns(2)

    with col1:
        st.session_state['selected_model'] = st.selectbox('Select a Model', options=[
                     'RandomForestClassifier', 'DecisionTreeClassifier'], key='select_model')
    
    with col2:
        pass
    
    if st.session_state['selected_model'] == 'RandomForestClassifier':
        pipeline = load_best_rf_model()
    else:
        pipeline = load_best_dt_model()
    
    encoder = joblib.load('./models/pipeline.joblib')

    return pipeline, encoder

# Preprocess input data
def preprocess_input_data(df):
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    # Apply one-hot encoding to categorical columns
    encoder = OneHotEncoder(drop='first')
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
    # Replace original categorical columns with the encoded columns
    df = pd.concat([df.drop(columns=categorical_cols), encoded_cols], axis=1)
    
    # df['Prediction Time'] = datetime.date.today()
    # df['Model Used'] = st.session_state['selected_model']
    
    
    # else:# Save data to CSV file
    # if not os.path.exists("./data/History.csv"):
    #         os.makedirs("./data")
    #         df.to_csv("./data/History.csv", index=False)
    #df.to_csv(".data/History/.csv", mode='a', header=False, index=False)

    
    
    
    # if not os.path.exists("./data/History.csv"):
    #     os.makedirs("./data/History.csv")
    df.to_csv("./data/History.csv", mode='a', header=not os.path.exists('./data/History.csv'), index=False)

    return df, encoder

# Make prediction
def make_prediction(pipeline, encoder):
    # Initialize session state variables
    st.session_state.setdefault('prediction', None)
    st.session_state.setdefault('probability', None)
    
    # Get input features
    gender = st.session_state['gender']
    seniorcitizen = st.session_state['seniorcitizen']
    partner = st.session_state['partner']
    dependents = st.session_state['dependents']
    tenure = st.session_state['tenure']
    phoneservice = st.session_state['phoneservice']
    multiplelines = st.session_state['multiplelines']
    internetservice = st.session_state['internetservice']
    onlinesecurity = st.session_state['onlinesecurity']
    onlinebackup = st.session_state['onlinebackup']
    deviceprotection = st.session_state['deviceprotection']
    techsupport = st.session_state['techsupport']
    streamingtv = st.session_state['streamingtv']
    streamingmovies = st.session_state['streamingmovies']
    contract = st.session_state['contract']
    paperlessbilling = st.session_state['paperlessbilling']
    paymentmethod = st.session_state['paymentmethod']
    monthlycharges = st.session_state['monthlycharges']
    totalcharges = st.session_state['totalcharges']

    # Create input data DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [seniorcitizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phoneservice],
        'MultipleLines': [multiplelines],
        'InternetService': [internetservice],
        'OnlineSecurity': [onlinesecurity],
        'OnlineBackup': [onlinebackup],
        'DeviceProtection': [deviceprotection],
        'TechSupport': [techsupport],
        'StreamingTV': [streamingtv],
        'StreamingMovies': [streamingmovies],
        'Contract': [contract],
        'PaperlessBilling': [paperlessbilling],
        'PaymentMethod': [paymentmethod],
        'MonthlyCharges': [monthlycharges],
        'TotalCharges': [totalcharges]
    })

    # Preprocess input data
    input_data_encoded = encoder.transform(input_data)

    # Make prediction
    prediction = pipeline.predict(input_data_encoded)
    probability = pipeline.predict_proba(input_data_encoded)

    # Update session state
    st.session_state['prediction'] = prediction
    st.session_state['probability'] = probability

    return prediction, probability

# Display form for input
def display_form():
    pipeline, encoder = select_model()

    with st.form('input-feature'):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('### **Demographics** ###')
            st.session_state['gender'] = st.radio('Gender', ["Male", "Female"])
            st.session_state['seniorcitizen'] = st.radio('Senior Citizen', ["No", "Yes"])
            st.session_state['partner'] = st.radio('Partner', ["Yes", "No"])
            st.session_state['dependents'] = st.radio('Dependents', ["Yes", "No"])
            st.session_state['tenure'] = st.number_input('Length of Tenure with the Telco (in months)', min_value=1, max_value=71, step=1)

        with col2:
            st.markdown('### **Charges** ###')
            st.session_state['monthlycharges'] = st.number_input('Monthly Charges', value=0.0)
            st.session_state['totalcharges'] = st.number_input('Total Charges', value=0.0)

        with col3:
            st.markdown('### **Billing Method** ###')
            st.session_state['contract'] = st.selectbox('Contract', ["Month-to-month", "One year", "Two year"])
            st.session_state['paperlessbilling'] = st.radio('Paperless Billing', ["Yes", "No"])
            st.session_state['paymentmethod'] = st.selectbox('Payment Method', ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col4:
            st.markdown('### **Services** ###')
            st.session_state['phoneservice'] = st.radio('Phone Service', ["Yes", "No"])
            st.session_state['multiplelines'] = st.selectbox('Multiple Lines', ["Yes", "No", "No phone service"])
            st.session_state['internetservice'] = st.selectbox('Internet Service', ["DSL", "Fiber optic", "No"])
            st.session_state['onlinesecurity'] = st.selectbox('Online Security', ["Yes", "No", "No internet service"])
            st.session_state['onlinebackup'] = st.selectbox('Online Backup', ["Yes", "No", "No internet service"])
            st.session_state['deviceprotection'] = st.selectbox('Device Protection', ["Yes", "No", "No internet service"])
            st.session_state['techsupport'] = st.selectbox('Tech Support', ["Yes", "No", "No internet service"])
            st.session_state['streamingtv'] = st.selectbox('Streaming TV', ["Yes", "No", "No internet service"])
            st.session_state['streamingmovies'] = st.selectbox('Streaming Movies', ["Yes", "No", "No internet service"])
        
        st.form_submit_button('Make your Prediction', on_click=make_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder))

# Batch prediction logic
def batch_prediction():
    upload_file = st.file_uploader("Upload CSV", type=["csv"])
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Update the path to the preprocessor model file
        preprocessor_path = os.path.join(script_directory, "..", "batch_models", "batch_pipeline_preprocessor.pkl")
        with open(preprocessor_path, 'rb') as file:
            preprocessor = joblib.load(file)
        X, y = preprocessor(df)
        model_option = st.selectbox("Select Model", ["SVC", "XGBoost"])
        if st.button("Predict"):
            if model_option == "SVC":
                svc_model_path = os.path.join(script_directory, "..", "batch_models", "svc_model.pkl")
                svc_model = joblib.load(svc_model_path)
                churn_percentage = predict(svc_model, X)  # Implement predict() function
            else:
                xgb_model_path = os.path.join(script_directory, "..", "models", "xgb_model.pkl")
                xgb_model = joblib.load(xgb_model_path)
                churn_percentage = predict(xgb_model, X)  # Implement predict() function
            st.success(f"Churn Percentage ({model_option} Model): {churn_percentage:.2f}%")

# Define predict function for batch predictions
def predict(model, X):
    # Implement prediction logic here
    pass

# Main function
def main():
    st.set_page_config(page_title='Churn Prediction', layout='wide')
    st.title('Customer Churn Predictor')
    
    prediction_option = st.radio("Select Prediction Option", ["Online", "Batch"])

    st.session_state.setdefault('prediction', None)
    st.session_state.setdefault('probability', None)

    if prediction_option == "Online":
        display_form()
    elif prediction_option == "Batch":
        batch_prediction()
    
    prediction = st.session_state['prediction']
    probability = st.session_state['probability']

    if prediction is None:
        st.markdown('### Predictions will show here')
    elif prediction == "Yes":
        probability_of_yes = probability[0][1] * 100
        st.markdown(f"### The Customer will Churn with a probability of {round(probability_of_yes, 2)}%")
                    
    else:
        probability_of_no = probability[0][0] * 100
        st.markdown(f"### The Customer will not Churn with a probability of {round(probability_of_no, 2)}% ")

    # # Display current prediction history
    # df = pd.read_csv("./data/History.csv")
    # st.write("Current Prediction History:", df)
    # Display session state
    st.write(st.session_state)
if __name__ == "__main__":
    main()
