# import streamlit as st

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

#     st.write("Please log in to proceed.")
 
#     # Login section
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     login_button = st.button("Login")
 
#     if login_button:
#         if username == "admin" and password == "password":
#             st.success("Login successful!")
#             # Redirect to other pages or display content for authenticated users
#         else:
#             st.error("Invalid username or password. Please try again.")
 
#     # Contact Information
#     st.subheader("Contact Information")
#     st.write("These are the links to the project's repository in github and articles on Medium and Linkedin")
#     st.markdown("- [Github](https://github.com/Ibrahim-I-Babana/Customer_Churn_Prediction_App.git) 🌍")
#     st.markdown("- [Linkedin](https://www.linkedin.com/in/babana-issahak-ibrahim/) 🌐")
#     st.markdown("- [Medium](https://medium.com/@lostangelib80) ✍")

# if __name__ == "__main__":
#     home_page()



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
#     col1, col2, col3 = st.columns([2, 2, 2])

#     with col1:
#         # Login and create account form
#         st.title("User Authentication")
#         session_state.form_state["username"] = st.text_input("Username", key=f"login_username_input_{generate_key()}", value=session_state.form_state["username"])
#         session_state.form_state["password"] = st.text_input("Password", type="password", key=f"login_password_input_{generate_key()}", value=session_state.form_state["password"])
#         login_button_key = generate_key()
#         login_button = st.button("Login", key=login_button_key)
#         session_state.form_state["new_username"] = st.text_input("New Username", key=f"new_username_input_{generate_key()}", value=session_state.form_state["new_username"])
#         session_state.form_state["new_password"] = st.text_input("New Password", type="password", key=f"new_password_input_{generate_key()}", value=session_state.form_state["new_password"])
#         session_state.form_state["confirm_password"] = st.text_input("Confirm Password", type="password", key=f"confirm_password_input_{generate_key()}", value=session_state.form_state["confirm_password"])
#         create_account_button_key = generate_key()
#         create_account_button = st.button("Create Account", key=create_account_button_key)

#         if create_account_button:
#             create_account()

#         if login_button:
#             login()

#     with col2:
#         # Display the home page content
#         home_page()

#     with col3:
#         # How to run the app
#         st.title("How to Run the App")
#         st.markdown("To run this app locally, make sure you have Python installed. Then, install Streamlit and the required dependencies by running the following command:")
#         st.code("pip install streamlit-authenticator")
#         st.markdown("After installing the dependencies, navigate to the directory containing the app file and run the following command:")
#         st.code("streamlit run app.py")

#         # Contact Information
#         st.title("Contact Information")
#         st.write("These are the links to the project's repository on GitHub and articles on Medium and LinkedIn:")
#         st.markdown("- [GitHub](https://github.com/Ibrahim-I-Babana/Customer_Churn_Prediction_App.git) 🌍")
#         st.markdown("- [LinkedIn](https://www.linkedin.com/in/babana-issahak-ibrahim/) 🌐")
#         st.markdown("- [Medium](https://medium.com/@lostangelib80) ✍")

# if __name__ == "__main__":
#     main()





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
# st.set_page_config(page_title="My Streamlit App", page_icon=":rocket:")

# # Initialize session state
# session_state = SessionState(authenticated=False, form_state={
#     "username": "",
#     "password": "",
#     "new_username": "",
#     "new_password": "",
#     "confirm_password": ""
# })

# # Home page function
# @st.cache_resource
# def home_page():
#     st.title("Welcome to My Streamlit App")
#     st.write("This is a Streamlit app created for demonstration purposes.")
#     st.write("Feel free to explore the features and functionalities!")

# # Logout function
# def logout():
#     session_state.authenticated = False

# # Login function
# @st.cache_data
# def login():
#     username = session_state.form_state["username"]
#     password = session_state.form_state["password"]
#     # Dummy authentication (replace with your own authentication logic)
#     if username == "admin" and password == "password":
#         session_state.authenticated = True
#     else:
#         st.error("Login failed: Incorrect username or password.")

# # Create Account function
# @st.cache_data
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
#     col1, col2, col3 = st.columns([4, 4, 6])

#     with col1:
#         st.title("My Streamlit App")
#         #st.sidebar.image("Images/rocket_icon.png", use_column_width=True)
#         session_state.form_state["username"] = st.text_input("Username", key=f"login_username_input_{generate_key()}", value=session_state.form_state["username"])
#         session_state.form_state["password"] = st.text_input("Password", type="password", key=f"login_password_input_{generate_key()}", value=session_state.form_state["password"])
#         login_button_key = generate_key()
#         login_button = st.button("Login", key=login_button_key)
#         session_state.form_state["new_username"] = st.text_input("New Username", key=f"new_username_input_{generate_key()}", value=session_state.form_state["new_username"])
#         session_state.form_state["new_password"] = st.text_input("New Password", type="password", key=f"new_password_input_{generate_key()}", value=session_state.form_state["new_password"])
#         session_state.form_state["confirm_password"] = st.text_input("Confirm Password", type="password", key=f"confirm_password_input_{generate_key()}", value=session_state.form_state["confirm_password"])
#         create_account_button_key = generate_key()
#         create_account_button = st.button("Create Account", key=create_account_button_key)

#         if create_account_button:
#             create_account()

#         if login_button:
#             login()

#     with col2:
#         # Display the home page content
#         home_page()

#     with col3:
#         # How to run the app
#         st.subheader("How to run the app:")
#         st.write("To run this app locally, make sure you have Python installed. Then, install Streamlit and the required dependencies by running the following command:")
#         st.code("pip install streamlit-authenticator")
#         st.write("After installing the dependencies, navigate to the directory containing the app file and run the following command:")
#         st.code("streamlit run app.py")

#         # Contact Information
#         st.subheader("Contact Information")
#         st.write("These are the links to the project's repository in GitHub and articles on Medium and LinkedIn")
#         st.markdown("- [GitHub](https://github.com/Ibrahim-I-Babana/Customer_Churn_Prediction_App.git) :earth_africa:")
#         st.markdown("- [LinkedIn](https://www.linkedin.com/in/babana-issahak-ibrahim/) :globe_with_meridians:")
#         st.markdown("- [Medium](https://medium.com/@lostangelib80) :memo:")

# if __name__ == "__main__":
#     main()


import streamlit_authenticator
import streamlit as st
import os
import yaml
from streamlit_authenticator import Authenticate
from Authenticate_Files.validator import Validator
from Authenticate_Files.utils import generate_random_pw
from session_state import SessionState
import uuid
from Authenticate_Files.hasher import Hasher

# Function to generate unique key
def generate_key():
    return str(uuid.uuid4())

# Load configuration from YAML file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# Set the favicon and title
st.set_page_config(page_title="Telecommunications Customer Churn Prediction App", page_icon="🚀")

# Initialize session state
session_state = SessionState(authenticated=False, form_state={
    "username": "",
    "password": "",
    "new_username": "",
    "new_password": "",
    "confirm_password": ""
})
    
    # Add image
st.image("Images/Global Telecommunications Network.jpg", caption="Telecommunications Logo", use_column_width=True)

# Logout function
def logout():
    session_state.authenticated = False

# Login function
def login():
    username = session_state.form_state["username"]
    password = session_state.form_state["password"]
    # Dummy authentication (replace with your own authentication logic)
    if username == "admin" and password == "password":
        session_state.authenticated = True
    else:
        st.error("Login failed: Incorrect username or password.")

# # Check if the user is authenticated
# if not st.session_state.get("authentication_status"):
#     st.info('Login from the Home page to use app')
# else:
#     #Set page title
#     st.markdown('### Proprietory Data from IBM 🛢️'



# with open('./config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)
 
# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )
 
 
# name, authentication_status, username = authenticator.login(location='sidebar')


# Create Account function
def create_account():
    new_username = session_state.form_state["new_username"]
    new_password = session_state.form_state["new_password"]
    confirm_password = session_state.form_state["confirm_password"]
    # Dummy account creation (replace with your own account creation logic)
    if new_password != confirm_password:
        st.error("Passwords do not match.")
    else:
        st.success(f"Account {new_username} created successfully!")

# def home_page():
#     st.title('Welcome to Telecommunications Customer Churn Prediction App')


# Main function
def main():
    st.markdown('### **Welcome to Telecommunications Customer Churn Prediction App** ###')
    # Login and create account form
    st.write("User Authentication")
    session_state.form_state["username"] = st.text_input("Username", key=f"login_username_input_{generate_key()}", value=session_state.form_state["username"])
    session_state.form_state["password"] = st.text_input("Password", type="password", key=f"login_password_input_{generate_key()}", value=session_state.form_state["password"])
    login_button_key = generate_key()
    login_button = st.button("Login", key=login_button_key)
    session_state.form_state["new_username"] = st.text_input("New Username", key=f"new_username_input_{generate_key()}", value=session_state.form_state["new_username"])
    session_state.form_state["new_password"] = st.text_input("New Password", type="password", key=f"new_password_input_{generate_key()}", value=session_state.form_state["new_password"])
    session_state.form_state["confirm_password"] = st.text_input("Confirm Password", type="password", key=f"confirm_password_input_{generate_key()}", value=session_state.form_state["confirm_password"])
    create_account_button_key = generate_key()
    create_account_button = st.button("Create Account", key=create_account_button_key)

   
    create_account()

    if login_button:
        login()

    # # Display the home page content
    # home_page()

    # How to run the app
    st.markdown("### **How to Run the App** ###")
    st.markdown("To run this app locally, make sure you have Python installed. Then, install Streamlit and the required dependencies by running the following command:")
    st.code("pip install streamlit-authenticator")
    st.markdown("After installing the dependencies, navigate to the directory containing the app file and run the following command:")
    st.code("streamlit run app.py")

    # Contact Information
    st.markdown("### **Contact Information** ###")
    st.write("These are the links to the project's repository on GitHub and articles on Medium and LinkedIn:")
    st.markdown("- [GitHub](https://github.com/Ibrahim-I-Babana/Customer_Churn_Prediction_App.git) 🌍")
    st.markdown("- [LinkedIn](https://www.linkedin.com/in/babana-issahak-ibrahim/) 🌐")
    st.markdown("- [Medium](https://medium.com/@lostangelib80) ✍")

if __name__ == "__main__":
    main()
