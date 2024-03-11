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


#     # data = [['gender', 'seniorcitizen', 'partner', 'dependents', 'tenure',
#     #                 'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
#     #                 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
#     #                 'contract', 'paperlessbilling', 'paymentmethod', 'monthlycharges', 'totalcharges']]
    

#     # columns = [['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
#     #    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
#     #    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
#     #    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
#     #    'MonthlyCharges', 'TotalCharges']]

    
#     # df = pd.DataFrame(data, columns = columns)


#     # #if not os.path.exists("./data/history.csv"):
#     #       #os.makedirs("./data/history.csv")
#     # df.to_csv("./data/history.csv",mode='a',header=not os.path.exists('./data/history.csv'),index=False)

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








 




# import os
# import joblib
# import pandas as pd
# import streamlit as st
# from sklearn.svm import SVC
# from xgboost import XGBClassifier

# def predict(model, X):
#     predictions = model.predict(X)
#     churn_percentage = (predictions.sum() / len(predictions)) * 100
#     return churn_percentage

# def batch_prediction():
#     upload_file = st.file_uploader("Upload CSV", type=["csv"])
#     if upload_file is not None:
#         df = pd.read_csv(upload_file)
#         script_directory = os.path.dirname(os.path.abspath(__file__))
#         preprocessor_path = os.path.join(script_directory, "models", "batch_pipeline_preprocessor.pkl")
#         with open(preprocessor_path, 'rb') as file:
#             preprocessor = joblib.load(file)
#         X, y = preprocessor(df)
#         model_option = st.selectbox("Select Model", ["SVC", "XGBoost"])
#         if st.button("Predict"):
#             if model_option == "SVC":
#                 svc_model_path = os.path.join(script_directory, "models", "svc_model.pkl")
#                 svc_model = joblib.load(svc_model_path)
#                 churn_percentage = predict(svc_model, X)
#             else:
#                 xgb_model_path = os.path.join(script_directory, "models", "xgb_model.pkl")
#                 xgb_model = joblib.load(xgb_model_path)
#                 churn_percentage = predict(xgb_model, X)
#             st.success(f"Churn Percentage ({model_option} Model): {churn_percentage:.2f}%")


 
# def main():
#     st.set_page_config(page_title="Churn Prediction", layout="wide")
#     st.title("Churn Prediction App")
 
#     prediction_option = st.radio("Select Prediction Option", ["Online", "Batch"])
 
#     if prediction_option == "Online":
#        online_prediction()
#     elif prediction_option == "Batch":
#         batch_prediction()
 
# if __name__ == "__main__":
#     main()



import streamlit as st
import pandas as pd
import joblib
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os

def predict(model, X):
    predictions = model.predict(X)
    churn_percentage = (predictions.sum() / len(predictions)) * 100
    return churn_percentage

def online_prediction():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('**Demographics**')
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])

    with col2:
        st.markdown('**Services**')
        phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No phone service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No phone service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No phone service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No phone service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No phone service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No phone service'])

    with col3:
        st.markdown('**Billing Method**')
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    with col4:
        st.markdown('**Charges**')
        monthly_charges = st.number_input('Monthly Charges', min_value=0)
        total_charges = st.number_input('Total Charges', min_value=0)
        tenure = st.number_input('Tenure', min_value=0)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        # Load the preprocessor
        preprocessor_path = os.path.join("models", "pipeline.joblib")
        with open(preprocessor_path, 'rb') as file:
            preprocessor = joblib.load(file)

        # Preprocess data
        preprocessed_data = preprocessor.transform(input_data)

        selected_model = st.session_state.get('model', 'DecisionTree')
        if selected_model == 'DecisionTree':
            # Load the Decision Tree model
            model_path = os.path.join("models", "best_dt_model.joblib")
            model = joblib.load(model_path)
            prediction = model.predict_proba(preprocessed_data)
        else:
            # Load the Random Forest model
            model_path = os.path.join("models", "best_rf_model.joblib")
            model = joblib.load(model_path)
            prediction = model.predict_proba(preprocessed_data)

        churn_percentage = prediction[0][1] * 100
        st.success(f'Churn Percentage ({selected_model} Model): {churn_percentage:.2f}%')

# def batch_prediction():
#     upload_file = st.file_uploader("Upload CSV", type=["csv"])
#     if upload_file is not None:
#         df = pd.read_csv(upload_file)
 
#         # Load the batch preprocessor
#         batch_preprocessor_path = os.path.join("models", "batch_pipeline_preprocessor.pkl")
#         with open(batch_preprocessor_path, 'rb') as file:
#             batch_preprocessor = joblib.load(file)
 
#         X = batch_preprocessor.transform(df)
 
#         model_option = st.selectbox("Select Model", ["SVC", "XGBoost"])
 
#         if model_option == "SVC":
#             # Load the SVC model
#             model_path = os.path.join("models", "svc_model.pkl")
#             model = joblib.load(model_path)
#             churn_percentage = predict(model, X)
#         else:
#             # Load the XGBoost model
#             model_path = os.path.join("models", "xgb_model.pkl")
#             model = joblib.load(model_path)
#             churn_percentage = predict(model, X)
 
#         st.success(f"Churn Percentage ({model_option} Model): {churn_percentage:.2f}%")




def batch_prediction():
    upload_file = st.file_uploader("Upload CSV", type=["csv"])
    if upload_file is not None:
        df = pd.read_csv(upload_file)

        # Load the preprocessor
        preprocessor_path = os.path.join("models", "batch_pipeline_preprocessor.pkl")
        with open(preprocessor_path, 'rb') as file:
            preprocessor = joblib.load(file)

        X, y = preprocessor(df)  # Apply the preprocessing pipeline directly to the DataFrame

        model_option = st.selectbox("Select Model", ["SVC", "XGBoost"])

        if model_option == "SVC":
            # Load the SVC model
            model_path = os.path.join("models", "svc_model.pkl")
            model = joblib.load(model_path)
            churn_percentage = predict(model, X)
        else:
            # Load the XGBoost model
            model_path = os.path.join("models", "xgb_model.pkl")
            model = joblib.load(model_path)
            churn_percentage = predict(model, X)

        st.success(f"Churn Percentage ({model_option} Model): {churn_percentage:.2f}%")




# def batch_prediction():
#     upload_file = st.file_uploader("Upload CSV", type=["csv"])
#     if upload_file is not None:
#         df = pd.read_csv(upload_file)

#         # Load the preprocessor
#         preprocessor_path = os.path.join("models", "batch_pipeline_preprocessor.pkl")
#         with open(preprocessor_path, 'rb') as file:
#             preprocessor = joblib.load(file)

#         X, y = preprocessor(df)

#         model_option = st.selectbox("Select Model", ["SVC", "XGBoost"])

#         if model_option == "SVC":
#             # Load the SVC model
#             model_path = os.path.join("models", "svc_model.pkl")
#             model = joblib.load(model_path)
#             churn_percentage = predict(model, X)
#         else:
#             # Load the XGBoost model
#             model_path = os.path.join("models", "xgb_model.pkl")
#             model = joblib.load(model_path)
#             churn_percentage = predict(model, X)

#         st.success(f"Churn Percentage ({model_option} Model): {churn_percentage:.2f}%")

def main():
    st.set_page_config(page_title="Churn Prediction", layout="wide")
    st.title("Churn Prediction App")

    prediction_option = st.radio("Select Prediction Option", ["Online", "Batch"])

    if prediction_option == "Online":
        st.session_state['model'] = st.selectbox('Select Model', ['DecisionTree', 'RandomForest'])
        online_prediction()
    elif prediction_option == "Batch":
        batch_prediction()

if __name__ == "__main__":
    main()