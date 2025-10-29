import streamlit as st
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Add the parent directory to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.models.classifier import SVMSpamClassifier
from src.data.dataset import SpamDataset

# Page config
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="✉️",
    layout="wide"
)

# Title
st.title("✉️ Spam Classifier Demo")

# Sidebar
st.sidebar.title("Settings")
threshold = st.sidebar.slider(
    "Prediction threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Adjust the classification threshold. Lower values increase recall but may reduce precision."
)

# Load data and model (do this only once)
@st.cache_resource
def load_resources():
    try:
        # Load model first
        model_path = Path("models/trained/spam_classifier.joblib")
        if not model_path.exists():
            raise FileNotFoundError("模型檔案未找到：" + str(model_path))

        # Load the model
        classifier, vectorizer, _ = SVMSpamClassifier.load_bundle(str(model_path))
        
        # Load dataset and get raw texts
        dataset = SpamDataset()
        try:
            X, y = dataset.load_data()
        except Exception as e:
            raise Exception("Failed to load dataset") from e
            
        X = dataset.preprocess_texts(X)
        
        # Split the raw texts first
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Now transform using the loaded vectorizer
        X_train = vectorizer.transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
        
        return (X_train, X_test, y_train, y_test), (classifier, vectorizer), X_test_raw
    except Exception as e:
        st.error(str(e))
        raise

try:
    (X_train, X_test, y_train, y_test), (classifier, vectorizer), raw_texts = load_resources()
    model_loaded = True
except Exception as e:
    st.error("Failed to load resources! Please ensure the data and model files exist.")
    st.error(f"Error: {str(e)}")
    st.exception(e)
    model_loaded = False
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["📝 Live Prediction", "📊 Model Analysis"])

# Tab 1: Real-time prediction
with tab1:
    st.header("Real-time Spam Detection")
    
    text_input = st.text_area(
        "Enter text to check:",
        height=100,
        placeholder="Enter email or SMS content here...",
    )

# Predict button and results
if st.button("Analyze", type="primary", disabled=not model_loaded):
    if not text_input.strip():
        st.warning("Please enter some text!")
    else:
        # Make prediction
        with st.spinner("Analyzing..."):
            X = vectorizer.transform([text_input])
            probability = classifier.predict_proba(X)[0][1]
            prediction = 1 if probability >= threshold else 0
            
            # Show results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("### Analysis Results")
                if prediction == 1:
                    st.error(f"⚠️ This is likely spam!")
                    st.warning("""
                    Recommended actions:
                    - ⛔ Do not click any links
                    - 🚫 Do not reply to the message
                    - ⚠️ Consider adding the sender to your block list
                    """)
                else:
                    st.success(f"✅ This appears to be legitimate mail")
                    
            with col2:
                st.write("### Predicted probability")
                
                # Create probability gauge
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie([probability, 1-probability], 
                      colors=['#ff4b4b' if probability >= threshold else '#919191', 
                             '#00cc96' if probability < threshold else '#919191'],
                      startangle=90)
                ax.add_artist(plt.Circle((0,0), 0.70, color='white'))
                plt.text(0, 0, f'{probability:.1%}', ha='center', va='center', fontsize=15)
                st.pyplot(fig)
                
                # Add legend
                st.markdown(f"Spam probability: **{probability:.1%}**")
                st.markdown(f"Threshold: **{threshold:.1%}**")

# Tab 2: Model Analysis
with tab2:
    st.header("Model Analysis Dashboard")
    
    # Class distribution
    st.subheader("Dataset Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=pd.DataFrame({'label': y_test}), x='label')
        plt.title("Test set class distribution")
        plt.xlabel("Class (0=Not Spam, 1=Spam)")
        plt.ylabel("Count")
        st.pyplot(fig)
    
    with col2:
        # Confusion Matrix
        y_pred = (classifier.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        st.pyplot(fig)
    
    # ROC and PR curves
    st.subheader("Model performance")
    col3, col4 = st.columns(2)
    
    with col3:
        # ROC Curve
        y_prob = classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        st.pyplot(fig)
    
    with col4:
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        st.pyplot(fig)

# About section
st.write("---")
st.write("### About this tool")
col_about1, col_about2 = st.columns(2)

with col_about1:
    st.write("""
    #### Technical highlights
    - 🤖 Uses a Support Vector Machine (SVM) classifier
    - 📊 Trained on over 5,000 real examples
    - 🎯 High precision and recall (reported on test set)
    - 📈 Adjustable prediction threshold
    """)

with col_about2:
    st.write("""
    #### Model performance (example)
    - ✅ Accuracy: 98.30%
    - 📋 Precision: 97.33%
    - 🎯 Recall: 90.68%
    - 💯 F1 score: 93.89%
    """)

# GitHub link
st.markdown("""
<div style='text-align: center'>
    <a href='https://github.com/howard92419/NCHU_Spam-Email' target='_blank'>
        🔍 View the code on GitHub
    </a>
</div>
""", unsafe_allow_html=True)