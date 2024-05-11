
import streamlit as st
# Define a function to display contact information of scientists
def connect_to_scientists():
    st.title("Connect to Scientists")
    st.write("Here are some scientists related to agricultural health:")

    # Display contact list of scientists
    scientists_list = [
        {"name": "Dr. John Doe", "email": "john@example.com", "phone": "123-456-7890"},
        {"name": "Dr. Jane Smith", "email": "jane@example.com", "phone": "987-654-3210"},
        # Add more scientists as needed
    ]

    for scientist in scientists_list:
        st.write(f"Name: {scientist['name']}, Email: {scientist['email']}, Phone: {scientist['phone']}")

# # Add this function call to your main app
# if page == "Connect to Scientists":
connect_to_scientists()
