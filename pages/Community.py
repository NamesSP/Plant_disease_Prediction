import streamlit as st
import praw

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="R5fQ44ayTqtWrr0M1VNHaQ",
    client_secret="Hcdlgsp0MRsDdyixk0nnTLgo9mRFig",
    user_agent="YOUR_USER_AGENT"
)

# Define a function to fetch and display Reddit posts
def display_reddit_posts(subreddit_name, num_posts):
    st.title(f"Community: r/{subreddit_name}")
    st.write("Here are some recent posts from the community:")

    # Fetch recent posts from the subreddit
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.new(limit=num_posts)

    # Display posts
    for post in posts:
        st.write(f"**{post.title}**")
        st.write(post.url)
        st.write(post.selftext)
        st.write("---")  # Add a separator between posts

# Define Streamlit app
def main():
    subreddit_name = st.text_input("Enter subreddit name (e.g., worldnews):")
    num_posts = st.slider("Number of posts to display:", min_value=1, max_value=20, value=10)

    if st.button("Fetch Posts"):
        if subreddit_name:
            display_reddit_posts(subreddit_name, num_posts)

# Run the Streamlit app
if __name__ == "__main__":
    main()
