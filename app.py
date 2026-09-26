import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.header('Image Classification Model')

model_path = r'C:\Users\HP\Desktop\Image_Classification\Image_classify.keras'

@st.cache_resource
def load_my_model():
    return load_model(model_path, compile=False)

model = load_my_model()

data_cat = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot',
            'cauliflower','chilli pepper','corn','cucumber','eggplant','garlic',
            'ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango','onion',
            'orange','paprika','pear','peas','pineapple','pomegranate','potato',
            'raddish','soy beans','spinach','sweetcorn','sweetpotato','tomato',
            'turnip','watermelon']

uploaded_file = st.file_uploader("Upload a Fruit/Vegetable Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image_load = Image.open(uploaded_file).resize((180, 180))
    st.image(image_load, caption='Uploaded Image', use_column_width=True)

    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    st.write(f'Veg/Fruit in image is: **{data_cat[np.argmax(score)]}**')
    st.write(f'Confidence: **{np.max(score)*100:.2f}%**')
