
# STITCHVISION AI

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
import pickle
import cv2
from pyembroidery import read_dst, JUMP

# PyTorch
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms

# DST generator
from dst import generate_dst

# Visualization
import matplotlib.pyplot as plt


# PAGE CONFIG

st.set_page_config(
    page_title="StitchVision AI",
    layout="wide",
    page_icon="🧵"
)

st.title("🧵 StitchVision AI — AI Embroidery Generator")


# TABS

tab1, tab2, tab3 = st.tabs([
    "📁 My Project",
    "📄 Project Description",
    "🧵 Embroidery Generator"
])


# EMBROIDERY TEXTURE CLASSES

EMB_CLASSES = [
    "braided","knitted","woven","lacelike","paisley",
    "striped","zigzagged","dotted","polka-dotted",
    "grid","crosshatched","interlaced"
]


# LOAD MODELS

@st.cache_resource
def load_texture_model():
    device = torch.device("cpu")
    checkpoint = torch.load("embroidery_texture_model.pth", map_location=device)

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(EMB_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device

@st.cache_resource
def load_svg_model():
    with open("SVG_MODEL.pkl", "rb") as f:
        model = pickle.load(f)
    return model

texture_model, texture_device = load_texture_model()
svg_model = load_svg_model()


# TRANSFORMS

torch_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# PREDICTION FUNCTIONS

def predict_texture(img_np):
    image = Image.fromarray(img_np).convert("RGB")
    image = torch_transform(image).unsqueeze(0).to(texture_device)

    with torch.no_grad():
        outputs = texture_model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return EMB_CLASSES[pred.item()], conf.item()

def generate_dotted(image_np):
    img = image_np.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = svg_model.predict(img, verbose=0)[0]
    if pred.ndim == 3:
        pred = np.mean(pred, axis=-1)

    dotted_mask = (pred > 0.5).astype(np.uint8) * 255
    dotted_path = "output_dotted.png"
    cv2.imwrite(dotted_path, dotted_mask)

    return dotted_mask, dotted_path


# DST VISUALIZATION

def visualize_dst(dst_path):
    pattern = read_dst(dst_path)
    lines = []
    x_pts, y_pts = [], []

    prev = None
    for s in pattern.stitches:
        x, y, cmd = s[0], -s[1], s[2]
        if cmd == JUMP:
            prev = None
            continue
        if prev is not None:
            lines.append(((prev[0], x), (prev[1], y)))
        prev = (x, y)
        x_pts.append(x)
        y_pts.append(y)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_facecolor("#fdfdfd")

    for lx, ly in lines:
        ax.plot(lx, ly, linewidth=0.8, alpha=0.8)

    ax.scatter(x_pts, y_pts, s=3)
    ax.axis("equal")
    ax.set_title(f"DST Stitch Preview | Total Stitches: {len(pattern.stitches)}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("dst_preview.png")
    plt.close()

    return "dst_preview.png"


# MY PROJECTS

with tab1:
    st.header("📁 My AI & Machine Learning Project")

    st.markdown("""
    ### 🧵 StitchVision AI (Flagship Project)
    **AI-powered system that converts images into industrial embroidery DST files**

    **Tech Stack**
    - Python, Streamlit
    - PyTorch (ResNet18)
    - TensorFlow (SVG Segmentation)
    - OpenCV, NumPy
    - DST Embroidery Format

    **Capabilities**
    - Texture prediction
    - Dotted embroidery mask generation
    - DST stitch file creation
    - Visual stitch preview
    """)

    st.markdown("""
    ---
    ### 🧠 Other Skills
    - Machine Learning (Supervised & Unsupervised)
    - Deep Learning 
    - Reinforcement Learning 
    """)


# PROJECT DESCRIPTION

with tab2:
    st.header("📄 StitchVision AI ")

    st.markdown("""
    ## 🔍 What is StitchVision AI?
    StitchVision AI is a **production-grade AI embroidery automation platform**
    that converts **normal images into machine-ready DST embroidery files**.

    It eliminates manual digitizing work using **AI + Computer Vision**.
    """)

    st.markdown("""
    ---
    ## ⚙️ How It Works

    1️⃣ Upload Image  
    2️⃣ AI predicts embroidery texture  
    3️⃣ Image converted to dotted stitch mask  
    4️⃣ Mask → DST embroidery file  
    5️⃣ Preview stitches before machine use
    """)

    st.markdown("""
    ---
    ## 🏭 Industry Use
    ✔ Garment Manufacturing  
    ✔ Textile Automation  
    ✔ Logo & Custom Embroidery  
    ✔ Fashion & Merchandising  

    **Result:** Faster production, lower cost, higher accuracy.
    """)

    st.markdown("""
    ---
    ## 🏭 Production Benefits

    - Faster embroidery digitization  
    - Consistent stitch quality  
    - Reduced human error  
    - Suitable for small & large embroidery units  
    """)



# EMBROIDERY GENERATOR

with tab3:
    uploaded_file = st.file_uploader(
        "📸 Upload Image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB").resize((256,256))
        image_np = np.array(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image_np, use_container_width=True)

        if st.button("🧵 Generate Embroidery"):
            st.session_state["dotted_mask"], st.session_state["dotted_path"] = generate_dotted(image_np)
            st.session_state["texture"], st.session_state["conf"] = predict_texture(image_np)
            st.session_state["dst_path"] = generate_dst(
                st.session_state["dotted_mask"],
                "StitchVision_Output.dst"
            )
            st.session_state["done"] = True

        if "done" in st.session_state:
            with col2:
                st.subheader("🧵 Dotted Design")
                st.image(st.session_state["dotted_path"], use_container_width=True)
                st.markdown(
                    f"**Texture:** {st.session_state['texture']}  \n"
                    f"**Confidence:** {st.session_state['conf']:.2f}"
                )

            with col3:
                st.subheader("🧵 DST Preview & Download")

                if st.checkbox("👁️ Preview DST"):
                    preview = visualize_dst(st.session_state["dst_path"])
                    st.image(preview, use_container_width=True)

                with open(st.session_state["dst_path"], "rb") as f:
                    st.download_button(
                        "⬇️ Download DST File",
                        data=f,
                        file_name="StitchVision_Output.dst",
                        mime="application/octet-stream"
                    )

            st.success("✅ Embroidery Generated Successfully!")
