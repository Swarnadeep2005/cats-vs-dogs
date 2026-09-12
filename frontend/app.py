"""
A minimal Streamlit demo UI: upload an image, see the prediction.

This talks to the FastAPI service over HTTP (not by importing
predict.py directly) on purpose — it's meant to demonstrate the real
client-server split you'd have in production, where a frontend and a
model-serving backend are separate deployable services. The API_URL
is read from an environment variable so the same code works whether
you're running this locally against `localhost:8000` or, inside
Docker Compose, against the `api` service by its container name.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐾")
st.title("🐾 Cats vs Dogs Classifier")
st.write(
    "Upload a photo and the model will predict whether it's a cat or a dog. "
    "Backed by a MobileNetV2 model fine-tuned via transfer learning."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the API at {API_URL}. "
                    f"Is the FastAPI service running?"
                )
            else:
                if response.status_code == 200:
                    result = response.json()
                    label = result["label"]
                    confidence = result["confidence"]

                    emoji = "🐱" if label == "cats" else "🐶"
                    st.success(f"{emoji} **{label.title()}** ({confidence:.1%} confidence)")
                else:
                    st.error(f"API error ({response.status_code}): {response.json().get('detail')}")

st.divider()
st.caption(f"API endpoint: {API_URL}")
