import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
import streamlit as st
from google import genai
from google.genai.errors import APIError
import gdown

# ==========================================================
# ⚙️ CONFIGURACIÓN
# ==========================================================

# 1. EL ID DE TU ARCHIVO EN GOOGLE DRIVE (¡ESTE SÍ PONLO AQUÍ!)
# Link ejemplo: drive.google.com/file/d/1A2b3C.../view -> ID: 1A2b3C...
GOOGLE_DRIVE_FILE_ID = "1qMLmfuY_LteFcruxGEuq-nqlIzVSmFNv"

# 2. Configuración del Modelo IA
GEMINI_MODEL_ID = "gemini-2.5-flash"
classes = ['ANEMIA', 'NO_ANEMIA']
path_modelo = 'best_model_ViT.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================================
# CARGA DEL MODELO (CON DESCARGA AUTOMÁTICA)
# ==========================================================
@st.cache_resource
def load_vit_model():
    # 1. Definir arquitectura
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(classes)
    )

    # 2. Descargar si no existe (Lógica de Nube)
    if not os.path.exists(path_modelo):
        # Validación para que no se te olvide el ID
        if "PEGA_AQUI" in GOOGLE_DRIVE_FILE_ID:
            st.error("❌ ERROR: Falta el ID de Google Drive en la línea 20 del código.")
            st.stop()

        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'

        print(f"🔽 Descargando modelo desde Drive...")
        st.info("☁️ Descargando modelo de IA desde la nube... (solo la primera vez)")

        try:
            gdown.download(url, path_modelo, quiet=False)
        except Exception as e:
            st.error(f"❌ Error al descargar de Drive: {e}")
            st.stop()

    # 3. Cargar pesos
    try:
        model.load_state_dict(torch.load(path_modelo, map_location=device))
    except RuntimeError as e:
        st.error(f"❌ Error de arquitectura: {e}")
        st.stop()

    model.to(device)
    model.eval()
    model.set_attn_implementation('eager')
    return model


# Inicializar modelo (se ejecuta una sola vez gracias a cache_resource)
model = load_vit_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ==========================================================
# FUNCIONES VISUALES
# ==========================================================
def generate_heatmap_and_transparency(att_map, grid_index, img, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = np.array(Image.fromarray(mask).resize((img.size), resample=Image.BILINEAR))
    mask = mask / np.max(mask)
    heatmap = Image.fromarray(np.uint8(plt.cm.rainbow(mask) * 255))
    heatmap_overlay = Image.blend(img.convert("RGBA"), heatmap, alpha=alpha)
    return heatmap_overlay


def concat_images_horizontally(img1, img2):
    w1, h1 = img1.size
    w2, h2 = img2.size
    if h1 != h2:
        img2 = img2.resize((int(w2 * h1 / h2), h1))
        w2, h2 = img2.size
    new_img = Image.new('RGB', (w1 + w2, h1))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (w1, 0))
    return new_img


# ==========================================================
# INTERFAZ DE USUARIO
# ==========================================================
st.set_page_config(page_title="Detector Anemia AI", page_icon="🩸")

st.title("🩸 Detector de Anemia + Gemini Flash")

# --- BARRA LATERAL SEGURA ---
with st.sidebar:
    st.header("🔑 Configuración")
    gemini_api_key = st.text_input("API Key de Google", type="password", help="Pega aquí tu clave de AI Studio")
    st.caption("La clave no se guarda, solo se usa para esta sesión.")

if not gemini_api_key:
    st.warning("👈 Por favor, ingresa tu API Key en el menú de la izquierda para continuar.")
else:
    uploaded_file = st.file_uploader("Sube imagen (JPG/PNG)", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        # --- ViT Inferencia ---
        with st.spinner("Analizando imagen..."):
            image_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(image_tensor, output_attentions=True)
                logits = preds.logits
                attention_maps = preds.attentions

            predicted_class = classes[torch.argmax(logits, dim=1).item()]

            heatmap_img = generate_heatmap_and_transparency(
                attention_maps[3][0, 0, 1:, 1:].cpu().detach().numpy(), 90, img
            )
            combined_img = concat_images_horizontally(img, heatmap_img)

        # --- Resultados ---
        st.success("Análisis completado.")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(label="Diagnóstico", value=predicted_class, delta="ViT AI")
        with col2:
            st.image(combined_img, caption="Original vs Atención", use_container_width=True)

        st.divider()

        # --- Gemini ---
        st.subheader("🤖 Interpretación Médica (Gemini)")

        if st.button("Generar Explicación"):
            prompt = (
                f"Actúa como oftalmólogo. Diagnóstico de IA: {predicted_class}. "
                f"La imagen derecha muestra zonas rojas donde la IA detectó patrones clave. "
                f"Analiza la palidez o vascularización en esas zonas y explica la correlación con el diagnóstico."
            )

            try:
                # Usamos la key que el usuario puso en el sidebar
                client = genai.Client(api_key=gemini_api_key)
                with st.spinner(f"Consultando a {GEMINI_MODEL_ID}..."):
                    response = client.models.generate_content(
                        model=GEMINI_MODEL_ID,
                        contents=[prompt, combined_img]
                    )
                st.markdown(response.text)

            except APIError as e:
                st.error(f"Error de Google API: {e}")
            except Exception as e:
                st.error(f"Error inesperado: {e}")