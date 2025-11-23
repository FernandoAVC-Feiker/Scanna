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
import gdown

# ==========================================================
# ⚙️ CONFIGURACIÓN DE LA PÁGINA
# ==========================================================
st.set_page_config(
    page_title="Scanna | AI Diagnosis",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed" # Colapsado por defecto en móvil
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #E63946;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #457B9D;
        text-align: center;
        margin-top: -5px;
        margin-bottom: 20px;
    }
    
    /* Ocultamos el menú hamburguesa genérico pero DEJAMOS el header 
       visible por si acaso, aunque ya no lo necesitamos tanto */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# ⚙️ CONFIGURACIÓN BACKEND
# ==========================================================

GEMINI_MODEL_ID = "gemini-2.5-flash" 
classes = ['ANEMIA', 'NO_ANEMIA']
path_modelo = 'best_model_ViT.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuración OOD
VIT_NAME = "google/vit-base-patch16-224-in21k"
MSP_THRESHOLD = 0.75
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
    # 1. Procesador
    try:
        processor = AutoImageProcessor.from_pretrained(VIT_NAME)
    except Exception as e:
        st.error(f"Error cargando procesador: {e}")
        st.stop()

    # 2. Modelo
    model = ViTForImageClassification.from_pretrained(
        VIT_NAME,
        num_labels=len(classes)
    )

    # 3. Verificación Local
    if not os.path.exists(path_modelo):
        st.error(f"❌ ERROR: No encuentro '{path_modelo}'")
        st.stop()

    # 4. Pesos
    try:
        model.load_state_dict(torch.load(path_modelo, map_location=device))
    except RuntimeError as e:
        st.error(f"❌ Error arquitectura: {e}")
        st.stop()

    model.to(device)
    model.eval()
    model.set_attn_implementation('eager')
    
    return model, processor

# Inicializar
model, processor = load_vit_system()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================================================
# LOGICA OOD
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
# FRONTEND (INTERFAZ MÓVIL OPTIMIZADA)
# ==========================================================

st.markdown('<p class="main-title">SCANNA</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Diagnóstico Oftalmológico IA</p>', unsafe_allow_html=True)

# --- ANIMACIÓN (Opcional en Sidebar o Main) ---
# En móvil, el sidebar está oculto, así que ponemos la animación pequeña arriba
lottie_eye = load_lottieurl("https://lottie.host/5f6d2c49-4f02-4476-9416-763167552060/l8W1Fw3Z4j.json")
if lottie_eye:
    st_lottie(lottie_eye, height=100, key="eye_mobile")

# --- CONFIGURACIÓN (EXPANDER PRINCIPAL) ---
# Esto reemplaza al sidebar para que sea accesible en celulares
with st.expander("🔐 Configuración y Acceso (Clic Aquí)", expanded=True):
    st.write("Ingresa tus credenciales para iniciar el sistema.")
    gemini_api_key = st.text_input("Google API Key", type="password", help="Tu clave de AI Studio")
    st.caption(f"Filtro de Calidad Activo: >{int(MSP_THRESHOLD*100)}% confianza")

# --- LÓGICA PRINCIPAL ---

if not gemini_api_key:
    st.info("👆 Por favor, despliega el menú de arriba e ingresa tu API Key.")
else:
    uploaded_file = st.file_uploader("📸 Sube la foto del ojo", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        # Feedback visual inmediato
        st.image(img, caption="Imagen cargada", use_container_width=True)
        
        my_bar = st.progress(0, text="Validando calidad...")
        
        # 1. FILTRO OOD
        time.sleep(0.2)
        is_valid, confidence, energy = check_image_quality(img, model, processor)
        
        if not is_valid:
            my_bar.empty()
            st.error("⛔ IMAGEN NO VÁLIDA")
            st.warning(f"Confianza muy baja ({confidence*100:.1f}%). La IA no reconoce esto como un ojo diagnosticable.")
            st.info("Intenta tomar la foto más cerca y con buena luz.")
            st.stop()
            
        # 2. INFERENCIA
        my_bar.progress(40, text="Analizando tejidos...")
        image_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(image_tensor, output_attentions=True)
            logits = preds.logits
            attention_maps = preds.attentions

        my_bar.progress(80, text="Generando mapas de calor...")
        predicted_class = classes[torch.argmax(logits, dim=1).item()]
        
        heatmap_img = generate_heatmap_and_transparency(
            attention_maps[3][0, 0, 1:, 1:].cpu().detach().numpy(), 90, img
        )
        combined_img = concat_images_horizontally(img, heatmap_img)
        
        my_bar.progress(100, text="¡Listo!")
        time.sleep(0.5)
        my_bar.empty()

        # 3. RESULTADOS (Diseño limpio)
        st.divider()
        st.subheader("Resultados")
        
        if predicted_class == "ANEMIA":
            st.error(f"⚠️ DETECCIÓN: {predicted_class}")
        else:
            st.success(f"✅ DETECCIÓN: {predicted_class}")
            
        st.caption("Izquierda: Original | Derecha: Zonas analizadas (Heatmap)")
        st.image(combined_img, use_container_width=True)

        st.divider()
        st.subheader("Opinión Médica (IA)")
        
        if st.button("📄 Generar Informe Detallado"):
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
                st.markdown(response.text)
            except APIError as e:
                st.error(f"Error API: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

