import streamlit as st

# Define a function to display products in a grid layout
def display_products():
    st.title("Buy Agricultural Products")
    st.write("Here are some agricultural products available for purchase:")

    # Define product details
    products = [
        {"name": "Product 1", "image": "https://via.placeholder.com/150", "price": "$10", "details": "Description of Product 1"},
        {"name": "Product 2", "image": "https://via.placeholder.com/150", "price": "$15", "details": "Description of Product 2"},
        {"name": "Product 3", "image": "https://via.placeholder.com/150", "price": "$20", "details": "Description of Product 3"},
        {"name": "Product 4", "image": "https://via.placeholder.com/150", "price": "$25", "details": "Description of Product 4"},
        {"name": "Product 5", "image": "https://via.placeholder.com/150", "price": "$30", "details": "Description of Product 5"},
        {"name": "Product 6", "image": "https://via.placeholder.com/150", "price": "$35", "details": "Description of Product 6"},
        {"name": "Product 7", "image": "https://via.placeholder.com/150", "price": "$40", "details": "Description of Product 7"},
        {"name": "Product 8", "image": "https://via.placeholder.com/150", "price": "$45", "details": "Description of Product 8"},
        {"name": "Product 9", "image": "https://via.placeholder.com/150", "price": "$50", "details": "Description of Product 9"},
        {"name": "Product 10", "image": "https://via.placeholder.com/150", "price": "$55", "details": "Description of Product 10"},
        {"name": "Product 11", "image": "https://via.placeholder.com/150", "price": "$60", "details": "Description of Product 11"},
        {"name": "Product 12", "image": "https://via.placeholder.com/150", "price": "$65", "details": "Description of Product 12"},
        {"name": "Product 13", "image": "https://via.placeholder.com/150", "price": "$70", "details": "Description of Product 13"},
        {"name": "Product 14", "image": "https://via.placeholder.com/150", "price": "$75", "details": "Description of Product 14"},
        {"name": "Product 15", "image": "https://via.placeholder.com/150", "price": "$80", "details": "Description of Product 15"},
        {"name": "Product 16", "image": "https://via.placeholder.com/150", "price": "$85", "details": "Description of Product 16"},
        {"name": "Product 17", "image": "https://via.placeholder.com/150", "price": "$90", "details": "Description of Product 17"},
        {"name": "Product 18", "image": "https://via.placeholder.com/150", "price": "$95", "details": "Description of Product 18"},
        {"name": "Product 19", "image": "https://via.placeholder.com/150", "price": "$100", "details": "Description of Product 19"},
        {"name": "Product 20", "image": "https://via.placeholder.com/150", "price": "$105", "details": "Description of Product 20"},
    ]

    # Display products in a grid layout
    num_columns = 5
    for i in range(0, len(products), num_columns):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            if i + j < len(products):
                cols[j].write(f"**{products[i + j]['name']}**")
                cols[j].image(products[i + j]['image'], use_column_width=True)
                cols[j].write(f"Price: {products[i + j]['price']}")
                cols[j].write(products[i + j]['details'])
                cols[j].write("---")  # Add a separator between products

# Display the products
display_products()
