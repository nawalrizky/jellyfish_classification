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
st.set_page_config(
    page_title="Jellyfish Classification", 
    layout="wide",
    page_icon="🪼",
    initial_sidebar_state="expanded"
)

# Add CSS for theme-responsive styling
st.markdown("""
<style>
:root {
    --background-color-secondary: rgba(240, 242, 246, 0.5);
    --border-color: rgba(49, 51, 63, 0.2);
    --text-color: rgba(49, 51, 63, 0.9);
}

[data-theme="dark"] {
    --background-color-secondary: rgba(38, 39, 48, 0.5);
    --border-color: rgba(250, 250, 250, 0.2);
    --text-color: rgba(250, 250, 250, 0.9);
}

/* Auto-detect dark mode */
@media (prefers-color-scheme: dark) {
    :root {
        --background-color-secondary: rgba(38, 39, 48, 0.5);
        --border-color: rgba(250, 250, 250, 0.2);
        --text-color: rgba(250, 250, 250, 0.9);
    }
}

/* Center content in columns - flexible height */
.stVerticalBlock.st-emotion-cache-vlxhtx.e1lln2w83,
.stHorizontalBlock.st-emotion-cache-ocqkz7.e1lln2w80 {
    display: flex !important;
    align-items: center !important;
    min-height: auto !important;
    height: auto !important;
}

.stHorizontalBlock.st-emotion-cache-ocqkz7.e1lln2w80 > div,
.stVerticalBlock.st-emotion-cache-vlxhtx.e1lln2w83 > div {
    width: 100%;
    height: auto !important;
}
</style>
""", unsafe_allow_html=True)

# Header with better styling
st.markdown("""
<div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #f0f8ff, #e6f3ff); border-radius: 10px; margin-bottom: 1rem;">
    <h1 style="color: #1f77b4; margin-bottom: 0; font-size: 2.5em;">🪼 Jellyfish Classification</h1>
    <p style="font-size: 1.2em; color: #666; margin-top: 0;">AI-powered jellyfish species identification</p>
    <p style="font-size: 0.9em; color: #888; margin-top: 0;">Upload a photo and let our AI identify the jellyfish species instantly</p>
</div>
""", unsafe_allow_html=True)

# Define classes - updated to match model's 6 output classes
CLASS_NAMES = [
    'barrel_jellyfish',
    'compass_jellyfish', 
    'lions_mane_jellyfish',
    'moon_jellyfish',
    'mauve_stinger_jellyfish',  # Added additional class
    'crystal_jellyfish'         # Added additional class
]

# Sidebar with information
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload** a clear image of a jellyfish
    2. **Wait** for the AI to analyze it
    3. **View** the prediction results
    
    **Supported species:**
    - 🪣 Barrel Jellyfish
    - 🧭 Compass Jellyfish  
    - 🦁 Lion's Mane Jellyfish
    - 🌙 Moon Jellyfish
    - 🟣 Mauve Stinger Jellyfish
    - 💎 Crystal Jellyfish
    """)
    
    st.markdown("### 📸 Image Tips")
    st.markdown("""
    - Use **clear, well-lit** photos
    - **Center** the jellyfish in frame
    - **Avoid** blurry or dark images
    - **JPG, PNG** formats supported
    """)
    
    st.markdown("---")
    st.markdown("*Powered by TensorFlow & Streamlit*")

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
    # Create two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Upload section
        st.markdown("### 📤 Upload Jellyfish Image")
        
        # Add helpful instructions
        st.markdown("""
        <div style="background-color: var(--background-color-secondary); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid var(--border-color);">
            <h4 style="margin-top: 0; color: var(--text-color);">📋 Quick Instructions:</h4>
            <ul style="margin-bottom: 0; color: var(--text-color);">
                <li>Choose a clear, well-lit photo</li>
                <li>Make sure the jellyfish is the main subject</li>
                <li>Accepted formats: JPG, PNG</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of a jellyfish for AI identification"
        )
    
    with col2:
        if uploaded_file is not None:
            # Display the uploaded image at true size
            img = Image.open(uploaded_file)
            st.image(img, caption='📸 Your uploaded image')
            
            # Show image info
            st.markdown(f"""
            <div style="background-color: var(--background-color-secondary); padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem; border: 1px solid var(--border-color);">
                <small style="color: var(--text-color);">📊 Image: {uploaded_file.name} ({img.size[0]}×{img.size[1]} pixels)</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("### 🖼️ Image Preview")
            st.info("Upload an image to see it displayed here")
    
    # Analysis results section (full width, no columns)
    if uploaded_file is not None:
        
        # Analysis and Results section
        st.markdown("---")  # Add separator
        
        with st.spinner('🔍 Analyzing your jellyfish image...'):
            try:
                # Get predictions
                predictions = predict_image(model, img)
                
                if predictions is not None:
                    # Get the predicted class and confidence score
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence_score = float(predictions[0][predicted_class_idx]) * 100
                    
                    # Display prediction results with visual feedback
                    st.markdown("### 🔍 Identification Results")
                    
                    # Main prediction with formatted name
                    predicted_name = CLASS_NAMES[predicted_class_idx].replace('_', ' ').title()
                    st.markdown(f"### **{predicted_name}**")
                    
                    # Confidence level with color coding and description
                    if confidence_score >= 80:
                        st.success(f"🎯 **High Confidence**: {confidence_score:.1f}%")
                        st.write("✅ The model is very confident about this identification.")
                    elif confidence_score >= 60:
                        st.warning(f"⚠️ **Medium Confidence**: {confidence_score:.1f}%")
                        st.write("🤔 The model has moderate confidence. Consider taking another photo.")
                    else:
                        st.error(f"❓ **Low Confidence**: {confidence_score:.1f}%")
                        st.write("🔄 The model is uncertain. Try a clearer photo or different angle.")
                    
                    # Show confidence as a progress bar
                    st.metric("Confidence Level", f"{confidence_score:.1f}%")
                    st.progress(confidence_score / 100)
                    
                    # Show top 3 predictions instead of all
                    with st.expander("🏆 Top 3 Most Likely Species"):
                        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                        
                        for i, idx in enumerate(top_3_indices):
                            species_name = CLASS_NAMES[idx].replace('_', ' ').title()
                            prob = float(predictions[0][idx]) * 100
                            
                            # Add medals for top 3
                            medals = ["🥇", "🥈", "🥉"]
                            col_left, col_right = st.columns([3, 1])
                            
                            with col_left:
                                st.write(f"{medals[i]} **{species_name}**")
                            with col_right:
                                st.write(f"{prob:.1f}%")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("💡 Try uploading a different image or check the file format.")
    else:
        st.markdown("### 👆 Upload an image to get started")
        st.info("🖼️ Select a jellyfish photo from your device using the upload button above.")

else:
    st.error("❌ **Model Loading Error**")
    st.markdown("""
    The AI model could not be loaded. This might be due to:
    - Missing model file
    - Corrupted model file
    - Insufficient memory
    
    Please try refreshing the page or contact support.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem 0; color: #666;">
    <p>🤖 <strong>About this AI:</strong> This model was trained to identify 6 different jellyfish species using deep learning.</p>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes. For accurate species identification, consult marine biology experts.</p>
    <p>📧 Questions or feedback? Feel free to reach out!</p>
</div>
""", unsafe_allow_html=True)
