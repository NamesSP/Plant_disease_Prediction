# import streamlit as st
# import cv2
# import tensorflow as tf
# import numpy as np
# import os

# # # Load the machine learning model
# # model = tf.keras.models.load_model('cat_dog_classifier.h5')

# # Define a function to upload the image file





# def upload_image():
#     uploaded_file = st.file_uploader("Upload an image")
#     if uploaded_file is not None:
#         # image = cv2.imread(uploaded_file)
#         image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
#         return image
    
# # Define a function to preprocess your images
# def preprocess_image(img):
#     # Load and preprocess the image using OpenCV or any other library
#     # Example: Read image, resize to model's input size, normalize pixel values
#     img = cv2.resize(img, (224, 224))  # Assuming model input size is 224x224
#     img = img / 255.0  # Normalize pixel values to [0, 1]
#     img = img.astype('float32')  # Convert to float32
#     return img


# def predict_class(img) :

#     # Make predictions using the loaded model
#     model = tf.saved_model.load(os.path.join(os.getcwd(),"archive"))
#     predictions = model(tf.constant([img]))
#     predicted_class = tf.argmax(predictions, axis=1)
#     res= predicted_class[0].numpy()

#     diseases = {
#     "0": "Apple___Apple_scab",
#     "1": "Apple___Black_rot",
#     "2": "Apple___Cedar_apple_rust",
#     "3": "Apple___healthy",
#     "4": "Blueberry___healthy",
#     "5": "Cherry_(including_sour)___Powdery_mildew",
#     "6": "Cherry_(including_sour)___healthy",
#     "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
#     "8": "Corn_(maize)___Common_rust_",
#     "9": "Corn_(maize)___Northern_Leaf_Blight",
#     "10": "Corn_(maize)___healthy",
#     "11": "Grape___Black_rot",
#     "12": "Grape___Esca_(Black_Measles)",
#     "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
#     "14": "Grape___healthy",
#     "15": "Orange___Haunglongbing_(Citrus_greening)",
#     "16": "Peach___Bacterial_spot",
#     "17": "Peach___healthy",
#     "18": "Pepper_bell___Bacterial_spot",
#     "19": "Pepper_bell___healthy",
#     "20": "Potato___Early_blight",
#     "21": "Potato___Late_blight",
#     "22": "Potato___healthy",
#     "23": "Raspberry___healthy",
#     "24": "Soybean___healthy",
#     "25": "Squash___Powdery_mildew",
#     "26": "Strawberry___Leaf_scorch",
#     "27": "Strawberry___healthy",
#     "28": "Tomato___Bacterial_spot",
#     "29": "Tomato___Early_blight",
#     "30": "Tomato___Late_blight",
#     "31": "Tomato___Leaf_Mold",
#     "32": "Tomato___Septoria_leaf_spot",
#     "33": "Tomato___Spider_mites Two-spotted_spider_mite",
#     "34": "Tomato___Target_Spot",
#     "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
#     "36": "Tomato___Tomato_mosaic_virus",
#     "37": "Tomato___healthy"
# }
#     return diseases[str(res)]






# # Create the Streamlit app
# st.title("PLANT DISEASE DETECTION")

# # Upload the image file
# image = upload_image()


# # If an image is uploaded, predict whether it contains a cat or a dog
# if image is not None:
#     image = preprocess_image(image) 

#     output = predict_class(image)

#     # # Display the prediction on the screen
#     #  display with more style and bigger font size
#     st.markdown("<h1 style='text-align: center; color: red;'>{}</h1>".format(output), unsafe_allow_html=True)

   

import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os

# Define a function to upload the image file
def upload_image():
    uploaded_file = st.file_uploader("Upload an image")
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        return image

# Define a function to preprocess your images
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Assuming model input size is 224x224
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = img.astype('float32')  # Convert to float32
    return img

# Define a function to predict the disease and associated treatments, medicines, and alternatives
def predict_class(img):
    model = tf.saved_model.load(os.path.join(os.getcwd(), "archive"))
    predictions = model(tf.constant([img]))
    predicted_class = tf.argmax(predictions, axis=1)
    res = predicted_class[0].numpy()

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
    
    treatments = {
        "Apple___Apple_scab": "1. Remove and destroy infected leaves to prevent the spread of spores.\n2. Apply fungicides containing copper or sulfur.\n3. Improve air circulation around plants by pruning.",
        "Apple___Black_rot": "1. Prune infected branches to remove affected areas.\n2. Apply fungicides before and after bloom to protect against infection.\n3. Remove fallen leaves and fruits to reduce overwintering spores.",
        "Apple___Cedar_apple_rust": "1. Remove and destroy galls on cedar trees.\n2. Apply fungicides to apple trees starting in spring before rust symptoms appear.\n3. Plant rust-resistant apple varieties.",
        "Apple___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Blueberry___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Cherry_(including_sour)___Powdery_mildew": "1. Apply fungicides containing sulfur or potassium bicarbonate.\n2. Prune infected branches to increase air circulation.\n3. Avoid overhead watering.",
        "Cherry_(including_sour)___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "1. Rotate crops to break disease cycles.\n2. Apply fungicides containing chlorothalonil or azoxystrobin.\n3. Remove and destroy infected plant debris.",
        "Corn_(maize)___Common_rust_": "1. Plant resistant corn varieties.\n2. Apply fungicides containing triazoles or strobilurins.\n3. Remove and destroy infected plants.",
        "Corn_(maize)___Northern_Leaf_Blight": "1. Rotate crops to break disease cycles.\n2. Apply fungicides containing chlorothalonil or azoxystrobin.\n3. Use balanced fertilization to promote plant health.",
        "Corn_(maize)___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Grape___Black_rot": "1. Apply fungicides containing captan or mancozeb.\n2. Remove and destroy infected plant parts.\n3. Prune vines to increase air circulation.",
        "Grape___Esca_(Black_Measles)": "1. Prune infected vines to remove diseased wood.\n2. Apply fungicides containing propiconazole or fosetyl-aluminum.\n3. Protect pruning wounds with wound dressings.",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "1. Apply fungicides containing captan or mancozeb.\n2. Remove and destroy infected leaves and debris.\n3. Prune vines to improve air circulation.",
        "Grape___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Orange___Haunglongbing_(Citrus_greening)": "1. Remove and destroy infected trees.\n2. Use insecticides to control psyllid vectors.\n3. Plant disease-free citrus nursery stock.",
        "Peach___Bacterial_spot": "1. Prune infected branches to remove cankers.\n2. Apply copper-based fungicides during the dormant season.\n3. Use drip irrigation to avoid wetting foliage.",
        "Peach___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Pepper_bell___Bacterial_spot": "1. Remove and destroy infected plants to prevent spread.\n2. Apply copper-based fungicides.\n3. Rotate crops to break disease cycles.",
        "Pepper_bell___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Potato___Early_blight": "1. Remove and destroy infected leaves and stems.\n2. Apply fungicides containing chlorothalonil or maneb.\n3. Plant disease-resistant potato varieties.",
        "Potato___Late_blight": "1. Apply fungicides containing chlorothalonil or maneb.\n2. Remove and destroy infected plants and tubers.\n3. Avoid overhead irrigation.",
        "Potato___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Raspberry___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Soybean___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Squash___Powdery_mildew": "1. Apply fungicides containing sulfur or potassium bicarbonate.\n2. Prune infected leaves to improve air circulation.\n3. Water at the base of plants to avoid wetting foliage.",
        "Strawberry___Leaf_scorch": "1. Remove and destroy infected leaves and plant debris.\n2. Apply fungicides containing copper or sulfur.\n3. Avoid overhead irrigation.",
        "Strawberry___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization.",
        "Tomato___Bacterial_spot": "1. Remove and destroy infected plants to prevent spread.\n2. Apply copper-based fungicides.\n3. Rotate crops to break disease cycles.",
        "Tomato___Early_blight": "1. Remove and destroy affected leaves.\n2. Apply fungicides containing chlorothalonil or copper hydroxide.\n3. Mulch around plants to reduce soil splashing onto leaves.",
        "Tomato___Late_blight": "1. Remove and destroy infected plants and fruits.\n2. Apply fungicides containing chlorothalonil or copper-based fungicides.\n3. Avoid overhead watering to reduce humidity levels.",
        "Tomato___Leaf_Mold": "1. Remove and destroy infected leaves.\n2. Apply fungicides containing chlorothalonil or mancozeb.\n3. Increase air circulation by spacing plants properly.",
        "Tomato___Septoria_leaf_spot": "1. Remove and destroy infected leaves.\n2. Apply fungicides containing chlorothalonil or copper hydroxide.\n3. Water plants at the base to avoid wetting foliage.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "1. Spray plants with water to dislodge mites.\n2. Apply miticides containing sulfur or neem oil.\n3. Introduce predatory mites to control spider mite populations.",
        "Tomato___Target_Spot": "1. Remove and destroy infected leaves.\n2. Apply fungicides containing chlorothalonil or mancozeb.\n3. Avoid overhead watering to reduce leaf wetness.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "1. Remove and destroy infected plants to prevent spread.\n2. Control whiteflies with insecticides.\n3. Plant virus-resistant tomato varieties.",
        "Tomato___Tomato_mosaic_virus": "1. Remove and destroy infected plants to prevent spread.\n2. Control aphid vectors with insecticides.\n3. Use virus-free planting material.",
        "Tomato___healthy": "No specific treatment needed. Maintain good cultural practices such as proper watering and fertilization."
    }
    
    medicines = {
        "Apple___Apple_scab": "1. Bordeaux mixture: Mix copper sulfate and hydrated lime in water. Apply every 10-14 days during the growing season.\n2. Neem oil: Dilute neem oil in water according to package instructions. Apply as a foliar spray to affected plants.",
        "Apple___Black_rot": "1. Bordeaux mixture: Mix copper sulfate and hydrated lime in water. Apply every 10-14 days during the growing season.\n2. Captan fungicide: Mix captan powder with water according to package instructions. Apply as a foliar spray.",
        "Apple___Cedar_apple_rust": "1. Mancozeb fungicide: Mix mancozeb powder with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Sulfur spray: Mix sulfur powder with water. Apply as a foliar spray every 7-10 days.",
        "Apple___healthy": "No specific plant medicines available.",
        "Blueberry___healthy": "No specific plant medicines available.",
        "Cherry_(including_sour)___Powdery_mildew": "1. Sulfur spray: Mix sulfur powder with water. Apply as a foliar spray every 7-10 days.\n2. Potassium bicarbonate spray: Mix potassium bicarbonate with water. Apply as a foliar spray every 7-10 days.",
        "Cherry_(including_sour)___healthy": "No specific plant medicines available.",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Rotate crops to break disease cycles.\n3. Use drip irrigation to avoid wetting foliage.",
        "Corn_(maize)___Common_rust_": "1. Triazole fungicide: Mix triazole fungicide with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Rotate crops to break disease cycles.\n3. Remove and destroy infected plant debris.",
        "Corn_(maize)___Northern_Leaf_Blight": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Rotate crops to break disease cycles.\n3. Use balanced fertilization to promote plant health.",
        "Corn_(maize)___healthy": "No specific plant medicines available.",
        "Grape___Black_rot": "1. Captan fungicide: Mix captan powder with water according to package instructions. Apply as a foliar spray.\n2. Mancozeb fungicide: Mix mancozeb powder with water according to package instructions. Apply as a foliar spray.",
        "Grape___Esca_(Black_Measles)": "1. Propiconazole fungicide: Mix propiconazole with water according to package instructions. Apply as a foliar spray every 10-14 days.\n2. Fosetyl-aluminum: Apply as a foliar spray or soil drench according to package instructions.",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "1. Captan fungicide: Mix captan powder with water according to package instructions. Apply as a foliar spray.\n2. Mancozeb fungicide: Mix mancozeb powder with water according to package instructions. Apply as a foliar spray.",
        "Grape___healthy": "No specific plant medicines available.",
        "Orange___Haunglongbing_(Citrus_greening)": "1. Remove and destroy infected trees.\n2. Apply insecticides to control psyllid vectors.\n3. Plant disease-resistant citrus varieties.",
        "Peach___Bacterial_spot": "1. Copper-based fungicide: Mix copper-based fungicide with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Streptomycin: Apply streptomycin according to package instructions.",
        "Peach___healthy": "No specific plant medicines available.",
        "Pepper_bell___Bacterial_spot": "1. Copper-based fungicide: Mix copper-based fungicide with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Rotate crops to break disease cycles.",
        "Pepper_bell___healthy": "No specific plant medicines available.",
        "Potato___Early_blight": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Remove and destroy infected leaves and stems.",
        "Potato___Late_blight": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Remove and destroy infected plants and tubers.",
        "Potato___healthy": "No specific plant medicines available.",
        "Raspberry___healthy": "No specific plant medicines available.",
        "Soybean___healthy": "No specific plant medicines available.",
        "Squash___Powdery_mildew": "1. Sulfur spray: Mix sulfur powder with water. Apply as a foliar spray every 7-10 days.\n2. Potassium bicarbonate spray: Mix potassium bicarbonate with water. Apply as a foliar spray every 7-10 days.",
        "Strawberry___Leaf_scorch": "1. Copper-based fungicide: Mix copper-based fungicide with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Remove and destroy infected leaves and plant debris.",
        "Strawberry___healthy": "No specific plant medicines available.",
        "Tomato___Bacterial_spot": "1. Copper-based fungicide: Mix copper-based fungicide with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Streptomycin: Apply streptomycin according to package instructions.",
        "Tomato___Early_blight": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Remove and destroy affected leaves.",
        "Tomato___Late_blight": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Remove and destroy infected plants and fruits.",
        "Tomato___Leaf_Mold": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Remove and destroy infected leaves.",
        "Tomato___Septoria_leaf_spot": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Remove and destroy infected leaves.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "1. Neem oil: Dilute neem oil in water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Insecticidal soap: Mix insecticidal soap with water according to package instructions. Apply as a foliar spray.",
        "Tomato___Target_Spot": "1. Chlorothalonil fungicide: Mix chlorothalonil with water according to package instructions. Apply as a foliar spray every 7-10 days.\n2. Remove and destroy infected leaves.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "1. Remove and destroy infected plants.\n2. Apply insecticides to control whitefly vectors.\n3. Use virus-free planting material.",
        "Tomato___Tomato_mosaic_virus": "1. Remove and destroy infected plants.\n2. Control aphid vectors with insecticides.\n3. Use virus-free planting material.",
        "Tomato___healthy": "No specific plant medicines available."
    }
    
    alternatives = {
        "Apple___Apple_scab": "1. Prune affected branches to improve air circulation.\n2. Use resistant apple varieties if available.\n3. Apply compost or organic mulch around the base of trees to promote soil health and plant immunity.",
        "Apple___Black_rot": "1. Improve drainage around trees to prevent waterlogged soil.\n2. Avoid overhead irrigation to reduce leaf wetness.\n3. Apply compost tea or seaweed extract as a foliar spray to boost plant immunity.",
        "Apple___Cedar_apple_rust": "1. Remove nearby cedar trees if possible to eliminate the alternate host.\n2. Prune apple trees to increase sunlight penetration and airflow.\n3. Apply compost or organic fertilizer to promote tree health.",
        "Apple___healthy": "No alternative treatments available.",
        "Blueberry___healthy": "No alternative treatments available.",
        "Cherry_(including_sour)___Powdery_mildew": "1. Improve air circulation by pruning branches.\n2. Plant disease-resistant cherry varieties if available.\n3. Apply horticultural oil to smother fungal spores.",
        "Cherry_(including_sour)___healthy": "No alternative treatments available.",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "1. Apply organic mulch to prevent soil splash.\n2. Maintain proper plant spacing to improve air circulation.\n3. Use drip irrigation to avoid wetting foliage.",
        "Corn_(maize)___Common_rust_": "1. Plant resistant corn varieties if available.\n2. Rotate crops to break disease cycles.\n3. Apply compost or organic fertilizer to promote plant health.",
        "Corn_(maize)___Northern_Leaf_Blight": "1. Apply balanced fertilization to improve plant health.\n2. Practice crop rotation to prevent buildup of pathogens.\n3. Use resistant corn hybrids if available.",
        "Corn_(maize)___healthy": "No alternative treatments available.",
        "Grape___Black_rot": "1. Prune grape vines to improve air circulation.\n2. Apply compost or organic mulch to promote soil health.\n3. Use resistant grape varieties if available.",
        "Grape___Esca_(Black_Measles)": "1. Prune infected grape vines to remove diseased wood.\n2. Apply horticultural oil to suffocate fungal spores.\n3. Apply compost or organic fertilizer to promote plant health.",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "1. Apply balanced fertilization to promote plant health.\n2. Prune grape vines to improve air circulation.\n3. Apply compost or organic mulch to suppress weeds and conserve moisture.",
        "Grape___healthy": "No alternative treatments available.",
        "Orange___Haunglongbing_(Citrus_greening)": "1. Plant disease-resistant citrus varieties if available.\n2. Apply organic mulch to suppress weeds and conserve soil moisture.\n3. Prune citrus trees to remove infected branches and improve air circulation.",
        "Peach___Bacterial_spot": "1. Apply compost tea to improve soil health.\n2. Prune peach trees to improve air circulation.\n3. Use resistant peach varieties if available.",
        "Peach___healthy": "No alternative treatments available.",
        "Pepper_bell___Bacterial_spot": "1. Apply compost tea to improve soil health.\n2. Use resistant pepper varieties if available.\n3. Practice crop rotation to prevent buildup of pathogens.",
        "Pepper_bell___healthy": "No alternative treatments available.",
        "Potato___Early_blight": "1. Apply compost or organic fertilizer to promote plant health.\n2. Rotate crops to break disease cycles.\n3. Use resistant potato varieties if available.",
        "Potato___Late_blight": "1. Apply compost tea to suppress soilborne pathogens.\n2. Prune potato plants to improve air circulation.\n3. Use drip irrigation to avoid wetting foliage.",
        "Potato___healthy": "No alternative treatments available.",
        "Raspberry___healthy": "No alternative treatments available.",
        "Soybean___healthy": "No alternative treatments available.",
        "Squash___Powdery_mildew": "1. Apply horticultural oil to suffocate fungal spores.\n2. Plant squash varieties resistant to powdery mildew if available.\n3. Apply compost or organic fertilizer to promote plant health.",
        "Strawberry___Leaf_scorch": "1. Apply compost tea to improve soil health.\n2. Use resistant strawberry varieties if available.\n3. Practice crop rotation to prevent disease buildup.",
        "Strawberry___healthy": "No alternative treatments available.",
        "Tomato___Bacterial_spot": "1. Apply compost or organic fertilizer to promote plant health.\n2. Use resistant tomato varieties if available.\n3. Practice crop rotation to break disease cycles.",
        "Tomato___Early_blight": "1. Apply compost tea to improve soil health.\n2. Use resistant tomato varieties if available.\n3. Mulch around plants to conserve soil moisture and suppress weeds.",
        "Tomato___Late_blight": "1. Apply compost or organic fertilizer to promote plant health.\n2. Use resistant tomato varieties if available.\n3. Practice crop rotation to prevent disease buildup.",
        "Tomato___Leaf_Mold": "1. Apply compost tea to improve soil health.\n2. Use resistant tomato varieties if available.\n3. Practice good garden sanitation to reduce disease spread.",
        "Tomato___Septoria_leaf_spot": "1. Apply compost or organic fertilizer to promote plant health.\n2. Use resistant tomato varieties if available.\n3. Mulch around plants to conserve soil moisture and suppress weeds.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "1. Apply neem oil to control spider mites.\n2. Introduce predatory mites to control spider mite populations.\n3. Apply insecticidal soap to suffocate spider mites.",
        "Tomato___Target_Spot": "1. Apply compost or organic fertilizer to promote plant health.\n2. Use resistant tomato varieties if available.\n3. Prune tomato plants to improve air circulation.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "1. Remove and destroy infected plants.\n2. Control whiteflies with insecticidal soap.\n3. Use reflective mulch to deter whiteflies.",
        "Tomato___Tomato_mosaic_virus": "1. Remove and destroy infected plants.\n2. Control aphid vectors with insecticidal soap.\n3. Use virus-free planting material.",
        "Tomato___healthy": "No alternative treatments available."
    }

    disease = diseases[str(res)]
    return disease, treatments[disease], medicines[disease], alternatives[disease]

# Create the Streamlit app
st.title("PLANT DISEASE DETECTION")

# Upload the image file
image = upload_image()

# If an image is uploaded, predict the disease and associated treatments, medicines, and alternatives
if image is not None:
    image = preprocess_image(image)
    disease, treatment, medicine, alternative = predict_class(image)

    # Display the detected disease
    st.subheader("Detected Disease")
    st.markdown("<h1 style='text-align: center; color: red;'>{}</h1>".format(disease), unsafe_allow_html=True)
    st.subheader("Treatments :")  
    st.write(treatment)
    st.subheader("Medicines :")
    st.write(medicine)   
    st.subheader("Alternatives :")       
    st.write(alternative)                                                                                    

    

