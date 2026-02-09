import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st
st.write("App started successfully")


# Load model
model = tf.keras.models.load_model("model.h5")  # or .keras

img_height = 224
img_width = 224

st.title("Image Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def preprocess_image(image):
    image = image.resize((img_height, img_width))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

        st.write("Confidence:", float(prediction))

        if prediction > 0.5:
            st.success("Rotten")
        else:
            st.success("Fresh")
