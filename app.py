# Requirements:
# pip install streamlit tensorflow pillow numpy pandas

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

# Set page title and configuration
st.set_page_config(page_title="Jellyfish Classification", layout="wide")
st.title("Jellyfish Classification App")

# Define classes - updated to match model's 6 output classes
CLASS_NAMES = [
    'barrel_jellyfish',
    'compass_jellyfish', 
    'lions_mane_jellyfish',
    'moon_jellyfish',
    'mauve_stinger_jellyfish',  # Added additional class
    'crystal_jellyfish'         # Added additional class
]

@st.cache_resource
def load_model_from_file():
    """Load the trained model"""
    try:
        model = load_model('model_jellyfish.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(img):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Resize image to 224x224 pixels
        img = img.resize((224, 224))
        # Convert to array and add batch dimension
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Normalize pixel values
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

def load_training_history():
    """Load training history if available"""
    try:
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        return history
    except:
        return None

def predict_image(model, img):
    """Process image and make prediction"""
    try:
        # Preprocess the image
        processed_img = preprocess_image(img)
        if processed_img is None:
            return None
            
      
        # Make prediction
        predictions = model.predict(processed_img)

        
        if len(predictions[0]) != len(CLASS_NAMES):
            st.error("Mismatch between model output classes and defined class names")
            return None
            
        return predictions
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Load the model
model = load_model_from_file()

# Main app
if model is not None:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image')
            
            # Get predictions
            predictions = predict_image(model, img)
            
            if predictions is not None:
                # Get the predicted class and confidence score
                predicted_class_idx = np.argmax(predictions[0])
                confidence_score = float(predictions[0][predicted_class_idx]) * 100
                
                # Display prediction results
                st.subheader("Prediction Results")
                st.write(f"Predicted Class: {CLASS_NAMES[predicted_class_idx]}")
                st.write(f"Confidence Score: {confidence_score:.2f}%")
                
                # Display probability distribution
                st.subheader("Probability Distribution")
                prob_dict = {class_name: float(prob) * 100 for class_name, prob in zip(CLASS_NAMES, predictions[0])}
                st.bar_chart(prob_dict)
                
                # Display training history if available
                history = load_training_history()
                if history is not None:
                    st.subheader("Training History")
                    
                    # Accuracy plot
                    st.line_chart({
                        'Training Accuracy': history['accuracy'],
                        'Validation Accuracy': history['val_accuracy']
                    })
                    
                    # Loss plot
                    st.line_chart({
                        'Training Loss': history['loss'],
                        'Validation Loss': history['val_loss']
                    })
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
else:
    st.error("Failed to load the model. Please check if the model file exists and is valid.")
