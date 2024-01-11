import os
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import base64
import json

# ------------------------------------
# Read constants from environment file
# ------------------------------------
load_dotenv()
APP_URI = os.environ.get("APP_URI")

# ------------------------------------
# Initialise Streamlit state variables
# ------------------------------------
def initialise_st_state_vars():
    if "auth_code" not in st.session_state:
        st.session_state["auth_code"] = ""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user_cognito_groups" not in st.session_state:
        st.session_state["user_cognito_groups"] = []

# ----------------------------------
# Set authorization code after login
# ----------------------------------
def set_auth_code():
    initialise_st_state_vars()
    auth_code = get_auth_code()
    st.session_state["auth_code"] = auth_code

# -----------------------------
# Login/ Logout HTML components
# -----------------------------
login_link = f"{APP_URI}/login"
logout_link = f"{APP_URI}/logout"

html_css_login = """
<style>
.button-login {
  background-color: skyblue;
  color: white !important;
  padding: 1em 1.5em;
  text-decoration: none;
  text-transform: uppercase;
}

.button-login:hover {
  background-color: #555;
  text-decoration: none;
}

.button-login:active {
  background-color: black;
}

</style>
"""

html_button_login = (
    html_css_login
    + f"<a href='{login_link}' class='button-login' target='_self'>Log In</a>"
)
html_button_logout = (
    html_css_login
    + f"<a href='{logout_link}' class='button-login' target='_self'>Log Out</a>"
)

def button_login():
    return st.sidebar.markdown(f"{html_button_login}", unsafe_allow_html=True)

def button_logout():
    return st.sidebar.markdown(f"{html_button_logout}", unsafe_allow_html=True)

# -------------------------------------------
# MongoDB Connection and User Authentication
# -------------------------------------------
def connect_to_mongodb(database_name='mcp'):
    client = MongoClient('mongodb://username:password@192.168.1.7:27017/')
    db = client[database_name]
    return client, db

def retrieve_data_from_collection(db, collection_name):
    collection = db[collection_name]
    return collection

def authenticate_user(username, password, login_collection):
    user_data = login_collection.find_one({"username": username, "password": password})
    return user_data is not None

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    client, db = connect_to_mongodb()
    login_collection = retrieve_data_from_collection(db, 'login_data')

    initialise_st_state_vars()

    if st.session_state.authenticated:
        st.write("You are logged in!")
        button_logout()
    else:
        button_login()
        st.sidebar.title("Login")

        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Log In"):
            if authenticate_user(username, password, login_collection):
                st.session_state.authenticated = True
                st.success("Successfully logged in!")
                set_auth_code()
                st.experimental_rerun()
            else:
                st.error("Invalid credentials. Please try again.")

if __name__ == "__main__":
    main()
