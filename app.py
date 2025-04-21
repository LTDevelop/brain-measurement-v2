import streamlit as st
import cv2
import numpy as np

st.set_page_config(layout="wide")
st.title("🧠 Medidor de Cérebros - Modo Avançado")

# Barra lateral com controles
with st.sidebar:
    st.header("⚙️ Parâmetros de Detecção")
    threshold = st.slider("Threshold de intensidade", 0, 255, 40, help="Ajuste para separar cérebros do fundo")
    min_area = st.slider("Área mínima (pixels)", 10, 1000, 100, help="Filtra objetos muito pequenos")
    blur = st.slider("Suavização (kernel)", 1, 15, 5, step=2, help="Reduz ruídos (ímpar)")

def processar_imagem(img, threshold, min_area, blur):
    # Pré-processamento
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (blur, blur), 0)
    
    # Detecção da régua (parte inferior)
    roi = cinza[-150:, :]
    _, bin_regua = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
    contornos_regua, _ = cv2.findContours(bin_regua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos_regua:
        st.warning("Régua não detectada! Posicione uma régua de 1cm na parte inferior.")
        return None
    
    # Escala (1cm = w pixels)
    x, y, w, h = cv2.boundingRect(max(contornos_regua, key=cv2.contourArea))
    escala = 1.0 / w  # cm por pixel

    # Segmentação dos cérebros (COR AJUSTÁVEL)
    _, bin_cerebros = cv2.threshold(cinza, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Operações morfológicas para limpar ruídos
    kernel = np.ones((3,3), np.uint8)
    bin_cerebros = cv2.morphologyEx(bin_cerebros, cv2.MORPH_OPEN, kernel)
    
    contornos, _ = cv2.findContours(bin_cerebros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra por área e proporção
    cerebros = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = w/h
        
        if area > min_area and 0.5 < aspect_ratio < 2.0:  # Filtra formatos não naturais
            cerebros.append(cnt)
    
    # Resultados
    img_resultado = img.copy()
    cv2.drawContours(img_resultado, cerebros, -1, (0, 255, 0), 2)
    
    # Desenha régua
    cv2.rectangle(img_resultado, (x, y+img.shape[0]-150), (x+w, y+h+img.shape[0]-150), (255,0,0), 2)
    
    return img_resultado, len(cerebros), escala

# Interface principal
upload = st.file_uploader("Carregue imagem:", type=["jpg","png"])

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, channels="BGR", caption="Imagem Original", use_column_width=True)
        
    with col2:
        resultado = processar_imagem(img, threshold, min_area, blur)
        
        if resultado:
            img_processada, num_cerebros, escala = resultado
            st.image(img_processada, channels="BGR", caption=f"Cérebros Detectados: {num_cerebros}", use_column_width=True)
            
            # Exibe métricas
            st.metric("Total de Cérebros", num_cerebros)
            st.metric("Escala", f"{escala:.4f} cm/pixel")
            
            # Botão de exportação
            st.download_button(
                "📥 Exportar Resultados",
                f"Cérebros detectados: {num_cerebros}\nEscala: {escala:.4f} cm/pixel",
                file_name="resultados.txt"
            )
