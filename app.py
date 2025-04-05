import streamlit as st
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
st.title("🔍 Duplicate Image Detection using CLIP + FAISS")
st.markdown("Upload an image and we will check if it's a duplicate based on visual similarity.")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = load_model()

# Initialize in-memory storage for embeddings
if "image_embeddings" not in st.session_state:
    st.session_state.image_embeddings = []
    st.session_state.image_paths = []

def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu().numpy().astype("float32")

def is_duplicate(new_embedding, threshold=0.9):
    if not st.session_state.image_embeddings:
        return False, None

    index = faiss.IndexFlatIP(512)
    index.add(np.array(st.session_state.image_embeddings))
    D, I = index.search(new_embedding, 1)
    similarity = D[0][0]
    return similarity > threshold, similarity

def store_embedding(embedding, name):
    st.session_state.image_embeddings.append(embedding[0])
    st.session_state.image_paths.append(name)

# Upload UI
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
threshold = st.slider("Similarity Threshold", 0.7, 1.0, 0.9, 0.01)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    embedding = get_image_embedding(image)

    duplicate, similarity = is_duplicate(embedding, threshold=threshold)

    if duplicate:
        st.error(f"⚠️ Duplicate detected (Similarity: {similarity:.2f})")
    else:
        st.success("✅ No duplicate found. Image added to memory.")
        store_embedding(embedding, uploaded_file.name)

# Show stored images if any
if st.session_state.image_paths:
    st.markdown("### Stored Images")
    st.write(st.session_state.image_paths)
