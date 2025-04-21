import streamlit as st
import cv2
import numpy as np

st.set_page_config(layout="wide")
st.title("🧠 Medidor de Cérebros - Modo Claro")

# Barra lateral com controles
with st.sidebar:
    st.header("⚙️ Ajustes de Detecção")
    threshold = st.slider("Limiar de brilho", 50, 250, 180, help="Valores mais altos = apenas áreas mais claras")
    min_area = st.slider("Área mínima (pixels)", 50, 500, 150, help="Filtra fragmentos pequenos")
    blur_size = st.slider("Suavização", 1, 15, 5, step=2, help="Reduz ruídos (use números ímpares)")

def processar_imagem(img, threshold, min_area, blur_size):
    # Converter para escala de cinza e suavizar
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (blur_size, blur_size), 0)
    
    # DETECÇÃO DE CÉREBROS CLAROS (mudança principal)
    _, bin_cerebros = cv2.threshold(cinza, threshold, 255, cv2.THRESH_BINARY)  # Agora pega áreas CLARAS
    
    # Operações morfológicas para limpeza
    kernel = np.ones((3,3), np.uint8)
    bin_cerebros = cv2.morphologyEx(bin_cerebros, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(bin_cerebros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por área e proporção
    cerebros = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = w/h
        
        if area > min_area and 0.3 < aspect_ratio < 3.0:  # Filtra formatos irregulares
            cerebros.append(cnt)
    
    # Detecção da régua (parte inferior)
    roi = cinza[-100:, :]
    _, bin_regua = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
    contornos_regua, _ = cv2.findContours(bin_regua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Preparar imagem de resultado
    img_resultado = img.copy()
    cv2.drawContours(img_resultado, cerebros, -1, (0, 255, 0), 2)
    
    if contornos_regua:
        x_r, y_r, w_r, h_r = cv2.boundingRect(max(contornos_regua, key=cv2.contourArea))
        cv2.rectangle(img_resultado, (x_r, y_r+img.shape[0]-100), (x_r+w_r, y_r+h_r+img.shape[0]-100), (255,0,0), 2)
        escala = 1.0 / w_r  # cm/pixel
    else:
        escala = None
    
    return img_resultado, len(cerebros), escala

# Interface principal
upload = st.file_uploader("Carregue imagem com cérebros claros e fundo escuro:", type=["jpg","png"])

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, channels="BGR", caption="Imagem Original", use_column_width=True)
        
    with col2:
        resultado = processar_imagem(img, threshold, min_area, blur_size)
        
        if resultado:
            img_processada, num_cerebros, escala = resultado
            st.image(img_processada, channels="BGR", caption=f"Cérebros Detectados: {num_cerebros}", use_column_width=True)
            
            st.success("✅ Ajuste os parâmetros na barra lateral para refinar a detecção")
            st.metric("Total de Cérebros", num_cerebros)
            
            if escala:
                st.metric("Escala", f"{escala:.4f} cm/pixel")
            else:
                st.warning("Régua não detectada - medidas em pixels")

# Dicas de uso
st.expander("💡 Dicas para melhor detecção").markdown("""
1. **Iluminação uniforme**: Evite sombras e reflexos
2. **Fundo escuro**: Cérebros devem ser mais claros que o fundo
3. **Régua preta**: Posicione na parte inferior
4. **Ajuste fino**:
   - Aumente o **limiar** para capturar apenas áreas muito claras
   - Aumente a **área mínima** para ignorar fragmentos
""")
