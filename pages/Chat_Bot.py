import streamlit as st
import openai

# Set OpenAI API key
openai.api_key = "sk-proj-5MKgWiMEoP5n6if2xDxVT3BlbkFJPjaV5wyy4kFAfLU0qlsn"

# Define function to interact with OpenAI's chat model
def chatbot(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",  # Adjust the model as needed
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {e}"

# Define Streamlit app
def main():
    st.title("Plant Disease Prediction Chatbot")
    st.write("Welcome to our interactive chatbot!")

    # Chat interface
    user_input = st.text_input("You: ")
    if st.button("Send"):
        if user_input:
            # Generate response from OpenAI
            bot_response = chatbot(user_input)
            st.text_area("Bot:", value=bot_response, height=100)

if __name__ == "__main__":
    main()
