import streamlit as st
import cv2
import numpy as np

st.set_page_config(layout="wide")
st.title("🧠 Medidor de Cérebros")

# Função para processamento
def processar_imagem(img):
    # 1. Converter para escala de cinza
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Detectar régua (parte inferior)
    roi = cinza[-100:, :]
    _, binario = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        st.error("⚠️ Régua não detectada! Posicione uma régua de 1cm na parte inferior.")
        return None
    
    # 3. Calcular escala (1cm = ? pixels)
    x, y, w, h = cv2.boundingRect(max(contornos, key=cv2.contourArea))
    escala = 1.0 / w  # 1cm por pixel
    
    # 4. Segmentar cérebros
    _, bin_cerebros = cv2.threshold(cinza, 30, 255, cv2.THRESH_BINARY)
    contornos_cerebros, _ = cv2.findContours(bin_cerebros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filtrar por tamanho (remover ruídos)
    cerebros = [cnt for cnt in contornos_cerebros if cv2.contourArea(cnt) > 100]
    
    # 6. Calcular área total (cm²)
    area_total = sum(cv2.contourArea(cnt) for cnt in cerebros) * (escala**2)
    
    # 7. Marcar na imagem
    img_resultado = img.copy()
    cv2.drawContours(img_resultado, cerebros, -1, (0, 255, 0), 3)
    cv2.rectangle(img_resultado, (x, y+img.shape[0]-100), (x+w, y+h+img.shape[0]-100), (255, 0, 0), 2)
    
    return img_resultado, area_total, len(cerebros)

# Interface
upload = st.file_uploader("Carregue imagem do cérebro com régua de 1cm visível:", type=["jpg", "png"])

if upload:
    arquivo = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(arquivo, cv2.IMREAD_COLOR)
    
    with st.spinner("Processando..."):
        resultado = processar_imagem(img)
    
    if resultado:
        img_final, area, num_celebros = resultado
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_final, channels="BGR", use_column_width=True)
        
        with col2:
            st.metric("Área Total", f"{area:.2f} cm²")
            st.metric("Número de Cérebros", num_celebros)
            
        st.download_button(
            "📥 Exportar Resultados",
            f"Área Total: {area:.2f} cm²\nCérebros Detectados: {num_celebros}",
            file_name="resultados.txt"
        )
