import os
import io
import time
import requests
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, AutoImageProcessor
import streamlit as st
from streamlit_lottie import st_lottie
from google import genai
from google.genai.errors import APIError
import gdown  # IMPORTANTE: Para bajar de Drive

# ==========================================================
# ⚙️ CONFIGURACIÓN DE LA PÁGINA
# ==========================================================
st.set_page_config(
    page_title="Scanna | AI Diagnosis",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #E63946;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #457B9D;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# ⚙️ CONFIGURACIÓN BACKEND
# ==========================================================

# TU ID DE DRIVE (Ya puesto)
GOOGLE_DRIVE_FILE_ID = "1qMLmfuY_LteFcruxGEuq-nqlIzVSmFNv"

GEMINI_MODEL_ID = "gemini-2.5-flash" 
classes = ['ANEMIA', 'NO_ANEMIA']
path_modelo = 'best_model_ViT.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuración OOD (Filtro de Calidad)
VIT_NAME = "google/vit-base-patch16-224-in21k"
MSP_THRESHOLD = 0.75  # 75% de confianza mínima
ENERGY_T = 2

# ==========================================================
# FUNCIONES DE CARGA
# ==========================================================

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

@st.cache_resource
def load_vit_system():
    # 1. Cargar Procesador (Desde HuggingFace)
    try:
        processor = AutoImageProcessor.from_pretrained(VIT_NAME)
    except Exception as e:
        st.error(f"Error cargando procesador: {e}")
        st.stop()

    # 2. Cargar Arquitectura
    model = ViTForImageClassification.from_pretrained(
        VIT_NAME,
        num_labels=len(classes)
    )

    # 3. DESCARGA DESDE DRIVE (Si no existe el archivo)
    if not os.path.exists(path_modelo):
        if "PEGA_AQUI" in GOOGLE_DRIVE_FILE_ID:
            st.error("❌ ERROR: ID de Drive inválido.")
            st.stop()
            
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
        
        # Mensaje de carga bonito
        with st.status("☁️ Descargando modelo desde la nube...", expanded=True) as status:
            st.write("Conectando con Google Drive...")
            try:
                gdown.download(url, path_modelo, quiet=False)
                st.write("¡Descarga completada!")
                status.update(label="Modelo listo", state="complete", expanded=False)
            except Exception as e:
                st.error(f"❌ Error al descargar de Drive: {e}")
                st.stop()

    # 4. Cargar Pesos
    try:
        model.load_state_dict(torch.load(path_modelo, map_location=device))
    except RuntimeError as e:
        st.error(f"❌ Error de arquitectura: {e}")
        st.stop()

    model.to(device)
    model.eval()
    model.set_attn_implementation('eager')
    
    return model, processor

# Inicializar sistema
model, processor = load_vit_system()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================================================
# VALIDACIÓN OOD (QUALITY CHECK)
# ==========================================================
def check_image_quality(pil_img, model, processor, threshold=MSP_THRESHOLD):
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=-1)[0]
    max_prob = float(probs.max().item())
    energy = float(-(ENERGY_T * torch.logsumexp(logits / ENERGY_T, dim=-1)).item())
    
    is_valid = max_prob >= threshold
    return is_valid, max_prob, energy

# ==========================================================
# FUNCIONES VISUALES
# ==========================================================
def generate_heatmap_and_transparency(att_map, grid_index, img, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple): grid_size = (grid_size, grid_size)
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
# FRONTEND
# ==========================================================

st.markdown('<p class="main-title">SCANNA</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Diagnóstico Oftalmológico Asistido por IA</p>', unsafe_allow_html=True)

with st.sidebar:
    lottie_eye = load_lottieurl("https://lottie.host/5f6d2c49-4f02-4476-9416-763167552060/l8W1Fw3Z4j.json")
    if lottie_eye:
        st_lottie(lottie_eye, height=150, key="eye")
    else:
        st.markdown("👁️ **SCANNA AI**")

    st.divider()
    st.header("🔐 Credenciales")
    gemini_api_key = st.text_input("Google API Key", type="password", help="Tu clave de AI Studio")
    
    st.info(f"**Filtro de Calidad Activo**\nConfianza mínima: {int(MSP_THRESHOLD*100)}%")

# --- LOGICA PRINCIPAL ---

if not gemini_api_key:
    st.warning("⚠️ Por favor, ingresa tu API Key en el menú lateral.")
    st.markdown(
        """<div style="text-align: center; color: gray; padding: 50px; border: 2px dashed #ccc; border-radius: 10px;">
            Esperando activación del sistema...
        </div>""", unsafe_allow_html=True
    )
else:
    uploaded_file = st.file_uploader("Sube la imagen del paciente (JPG/PNG)", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        my_bar = st.progress(0, text="Validando calidad de imagen...")
        
        # 1. FILTRO OOD
        time.sleep(0.2)
        is_valid, confidence, energy = check_image_quality(img, model, processor)
        
        if not is_valid:
            my_bar.empty()
            st.error("⛔ IMAGEN RECHAZADA: Calidad insuficiente o fuera de distribución.")
            col1, col2 = st.columns(2)
            with col1: st.image(img, caption="Imagen", use_container_width=True)
            with col2:
                st.metric("Confianza", f"{confidence*100:.1f}%")
                st.metric("Mínimo Requerido", f"{MSP_THRESHOLD*100:.0f}%")
                st.warning("La IA no reconoce esto como un ojo válido para diagnóstico.")
            st.stop()
            
        # 2. INFERENCIA
        my_bar.progress(30, text="Imagen válida. Procesando...")
        image_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(image_tensor, output_attentions=True)
            logits = preds.logits
            attention_maps = preds.attentions

        my_bar.progress(70, text="Generando mapas de calor...")
        predicted_class = classes[torch.argmax(logits, dim=1).item()]
        
        heatmap_img = generate_heatmap_and_transparency(
            attention_maps[3][0, 0, 1:, 1:].cpu().detach().numpy(), 90, img
        )
        combined_img = concat_images_horizontally(img, heatmap_img)
        
        my_bar.progress(100, text="¡Análisis Completado!")
        time.sleep(0.5)
        my_bar.empty()

        # 3. RESULTADOS
        tab1, tab2 = st.tabs(["📊 Diagnóstico Visual", "📝 Informe Médico (IA)"])

        with tab1:
            st.subheader("Resultados del Modelo ViT")
            st.caption(f"✅ Calidad aprobada (Confianza: {confidence*100:.1f}%)")
            
            if predicted_class == "ANEMIA":
                st.error(f"⚠️ DIAGNÓSTICO POSITIVO: {predicted_class}")
            else:
                st.success(f"✅ DIAGNÓSTICO NEGATIVO: {predicted_class}")
            
            st.image(combined_img, caption="Original vs Atención IA", use_container_width=True)

        with tab2:
            st.subheader("Análisis Fisiopatológico con Gemini")
            if st.button("✨ Generar Explicación Detallada", type="primary"):
                # PROMPT ORIGINAL
                prompt = (
                    f"Analiza la Imagen A (entrada cruda) y la Imagen B (mapa de atención asociado). "
                    f"Las siguientes imágenes pertenecen a la clase {predicted_class} según el clasificador de anemia. "
                    f"Explica en un solo párrafo qué regiones resaltadas en B guiaron la decisión, "
                    f"qué rasgos visuales en A (color, vascularización, textura o palidez) sustentan la pertenencia a {predicted_class}, "
                    f"y cómo estos se relacionan fisiológicamente con la presencia o ausencia de anemia. "
                    f"Mantén la explicación breve, médica y directamente basada en lo que se observa."
                )

                try:
                    client = genai.Client(api_key=gemini_api_key)
                    with st.spinner(f"Consultando a {GEMINI_MODEL_ID}..."):
                        response = client.models.generate_content(
                            model=GEMINI_MODEL_ID,
                            contents=[prompt, combined_img]
                        )
                    st.markdown("### 💬 Reporte:")
                    st.info(response.text)
                except APIError as e:
                    st.error(f"Error de Google API: {e}")
                except Exception as e:
                    st.error(f"Error inesperado: {e}")

