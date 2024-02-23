# import streamlit as st
 
# def home_page():
#     st.title('Welcome to Telecommunications Customer Churn Prediction App')
#     st.write("Losing customers is a major cost for any organization.")
#     st.write("Customer churn, also known as customer attrition or customer turnover, is the percentage of customers who stop using your company’s product or service within a certain period of time.") 
#     st.write("For example, if you started the year with 500 customers and ended with 480 customers, your customer churn rate would be 4%.")
#     st.write("If we could understand the reasons and the timing of customer churn with reasonable accuracy, it would greatly help the organization to plan and implement effective retention strategies.")

#     st.write("This is a Classification project aimed at assisting a telecommunications company understand their data and find the life time value of each customer and know what factors affect the rate at which customers stop using their network.")
#     st.write("The predictive modelling was done after a comprehensive analysis of the dataset provided by the business team")

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
 
# if __name__ == "__main__":
#     home_page()

# st.subheader("Contact Information")
# st.write("These are the links to the project's repository in github")
# st.page_link("https://github.com/Ibrahim-I-Babana/Customer_Churn_Prediction_App.git", label="Github", icon="🌍")
# st.page_link("https://www.linkedin.com/in/babana-issahak-ibrahim/", label="Linkedin", icon="🌐")
# st.page_link("https://medium.com/@lostangelib80", label="Medium", icon="✍")

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
#     st.image("C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\Images\Global Telecommunications Network.jpg", caption="Telecommunications Logo", use_column_width=True)

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

import streamlit as st

def home_page():
    st.title('Welcome to Telecommunications Customer Churn Prediction App')
    st.markdown("""
    Losing customers is a major cost for any organization. Customer churn, also known as customer attrition or customer turnover, is the percentage of customers who stop using your company’s product or service within a certain period of time.
    
    For example, if you started the year with 500 customers and ended with 480 customers, your customer churn rate would be 4%.
    
    If we could understand the reasons and the timing of customer churn with reasonable accuracy, it would greatly help the organization to plan and implement effective retention strategies.

    This is a Classification project aimed at assisting a telecommunications company understand their data and find the life time value of each customer and know what factors affect the rate at which customers stop using their network.
    """)
    
    # Add image
    st.image(r"C:\Users\user\OneDrive\Desktop\MY DS CAREER ACCELERATOR\Customer_Churn_Prediction_App\Images\Global Telecommunications Network.jpg", caption="Telecommunications Logo", use_column_width=True)

    st.write("Please log in to proceed.")
 
    # Login section
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
 
    if login_button:
        if username == "admin" and password == "password":
            st.success("Login successful!")
            # Redirect to other pages or display content for authenticated users
        else:
            st.error("Invalid username or password. Please try again.")
 
    # Contact Information
    st.subheader("Contact Information")
    st.write("These are the links to the project's repository in github and articles on Medium and Linkedin")
    st.markdown("- [Github](https://github.com/Ibrahim-I-Babana/Customer_Churn_Prediction_App.git) 🌍")
    st.markdown("- [Linkedin](https://www.linkedin.com/in/babana-issahak-ibrahim/) 🌐")
    st.markdown("- [Medium](https://medium.com/@lostangelib80) ✍")

if __name__ == "__main__":
    home_page()

