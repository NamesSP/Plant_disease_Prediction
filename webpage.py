import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

# # Load the machine learning model
# model = tf.keras.models.load_model('cat_dog_classifier.h5')

# Define a function to upload the image file





def upload_image():
    uploaded_file = st.file_uploader("Upload an image")
    if uploaded_file is not None:
        # image = cv2.imread(uploaded_file)
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        return image
    
# Define a function to preprocess your images
def preprocess_image(img):
    # Load and preprocess the image using OpenCV or any other library
    # Example: Read image, resize to model's input size, normalize pixel values
    img = cv2.resize(img, (224, 224))  # Assuming model input size is 224x224
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = img.astype('float32')  # Convert to float32
    return img


def predict_class(img) :

    # Make predictions using the loaded model
    model = tf.saved_model.load(".\\archive\\")
    predictions = model(tf.constant([img]))
    predicted_class = tf.argmax(predictions, axis=1)
    res= predicted_class[0].numpy()

    diseases = {
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy",
    "4": "Blueberry___healthy",
    "5": "Cherry_(including_sour)___Powdery_mildew",
    "6": "Cherry_(including_sour)___healthy",
    "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "8": "Corn_(maize)___Common_rust_",
    "9": "Corn_(maize)___Northern_Leaf_Blight",
    "10": "Corn_(maize)___healthy",
    "11": "Grape___Black_rot",
    "12": "Grape___Esca_(Black_Measles)",
    "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "14": "Grape___healthy",
    "15": "Orange___Haunglongbing_(Citrus_greening)",
    "16": "Peach___Bacterial_spot",
    "17": "Peach___healthy",
    "18": "Pepper_bell___Bacterial_spot",
    "19": "Pepper_bell___healthy",
    "20": "Potato___Early_blight",
    "21": "Potato___Late_blight",
    "22": "Potato___healthy",
    "23": "Raspberry___healthy",
    "24": "Soybean___healthy",
    "25": "Squash___Powdery_mildew",
    "26": "Strawberry___Leaf_scorch",
    "27": "Strawberry___healthy",
    "28": "Tomato___Bacterial_spot",
    "29": "Tomato___Early_blight",
    "30": "Tomato___Late_blight",
    "31": "Tomato___Leaf_Mold",
    "32": "Tomato___Septoria_leaf_spot",
    "33": "Tomato___Spider_mites Two-spotted_spider_mite",
    "34": "Tomato___Target_Spot",
    "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "36": "Tomato___Tomato_mosaic_virus",
    "37": "Tomato___healthy"
}
    return diseases[str(res)]






# Create the Streamlit app
st.title("PLANT DISEASE DETECTION")

# Upload the image file
image = upload_image()


# If an image is uploaded, predict whether it contains a cat or a dog
if image is not None:
    image = preprocess_image(image) 

    output = predict_class(image)

    # # Display the prediction on the screen
    #  display with more style and bigger font size
    st.markdown("<h1 style='text-align: center; color: red;'>{}</h1>".format(output), unsafe_allow_html=True)
