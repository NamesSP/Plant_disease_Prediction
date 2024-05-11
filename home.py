import streamlit as st 
from streamlit_extras.switch_page_button import switch_page


def diseasedetect():
   switch_page("disease")

def scientistconnect() :
   switch_page("Connect_To_Scientist")

def chatbot() :
    switch_page("Chat_Bot")

def community() :
   switch_page("Community")

def agriproducts() :
    switch_page("Buy_Agri_Products")

st.title("HOME PAGE ")

# Create three columns
col1= st.columns([1])[0]

st.write("")
st.write("")
st.write("")


st.markdown("*******")


col2, col3,col4, col5 = st.columns([1,1,1,1])

with col1:
  st.title("Connect to Scientist")
  st.button("CLICK HERE!",on_click=scientistconnect,key="b1")
  

with col2:
  st.write("Detect disease")
  st.button("CLICK HERE!",on_click=diseasedetect,key="b2") 

with col3:
  st.write("Chat Bot")
  st.button("CLICK HERE!",on_click=chatbot,key="b3")

with col4:
    st.write("Community ")
    st.button("CLICK HERE!",on_click=community,key="b4")

with col5 :
   st.write("Agriculture Products ")
   st.button("CLICK HERE!",on_click=agriproducts,key="b6")
