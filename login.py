import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def switcher() :
   switch_page("home")

def login(username, password):
  # Check if the username and password are valid
  if username == "admin" and password == "password":
    # Login successful
    st.session_state["authenticated"] = True
    st.session_state["username"] = username
  else:
    # Login failed
    st.session_state["authenticated"] = False 
    st.error("Invalid username or password")


# Add a title and subtitle to the login page
st.title("Login Page")
st.write("Please enter your username and password to login")

# Create a login form with two input fields for username and password
username = st.text_input("Username:",)
password = st.text_input("Password:", type="password")

# Add a button to submit the login form
if st.button("Login"):
  login(username, password)
    # If the user is authenticated, redirect to the home page
  if st.session_state["authenticated"]:
        
        st.write("Successfully logged in!")
        st.write(f"Welcome, {st.session_state['username']}!")

        # Redirect to the home page

        switcher()

        # Add your home page content here
