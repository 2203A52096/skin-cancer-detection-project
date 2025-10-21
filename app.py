import streamlit as st

st.set_page_config(
    page_title="Safe Skin",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Welcome to Safe Skin")
st.write("""
**Safe Skin** is an AI-based system for **early detection of skin cancer**.  

Navigate using the sidebar:  
- **Home:** Project overview, goal, and advantages  
- **Summary:** Dataset details, classes, and workflow  
- **Prediction:** Upload dermoscopic images and get predictions  
""")
