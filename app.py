import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧠 Medidor de Cérebros - Modo Profissional")

# Barra lateral
with st.sidebar:
    st.header("⚙️ Controles")
    threshold = st.slider("Limiar de brilho", 50, 250, 180)
    min_area_cm = st.slider("Área mínima (cm²)", 0.1, 2.0, 0.5, step=0.1)
    blur_size = st.slider("Suavização", 1, 15, 5, step=2)

def processar_imagem(img, threshold, min_area_cm, blur_size):
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (blur_size, blur_size), 0)
    
    # Detecção da régua
    roi = cinza[-150:, :]
    _, bin_regua = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
    contornos_regua, _ = cv2.findContours(bin_regua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos_regua:
        st.warning("Posicione uma régua de 1cm na base da imagem")
        return None
    
    # Cálculo da escala
    x_r, y_r, w_r, h_r = cv2.boundingRect(max(contornos_regua, key=cv2.contourArea))
    escala = 1.0 / w_r  # cm/pixel
    min_area_px = min_area_cm / (escala ** 2)  # Converte cm² para pixels

    # Detecção dos cérebros
    _, bin_cerebros = cv2.threshold(cinza, threshold, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(bin_cerebros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Processamento dos resultados
    resultados = []
    img_resultado = img.copy()
    
    for i, cnt in enumerate(contornos):
        area_px = cv2.contourArea(cnt)
        area_cm = area_px * (escala ** 2)
        
        if area_cm >= min_area_cm:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.drawContours(img_resultado, [cnt], -1, (0,255,0), 2)
            cv2.putText(img_resultado, f"{i+1}", (x,y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            resultados.append({
                "Cérebro": i+1,
                "Área (cm²)": round(area_cm, 2),
                "Centroide (x,y)": (x + w//2, y + h//2)
            })

    # Desenha régua
    cv2.rectangle(img_resultado, (x_r, y_r+img.shape[0]-150), 
                 (x_r+w_r, y_r+h_r+img.shape[0]-150), (255,0,0), 2)
    
    return img_resultado, pd.DataFrame(resultados), escala

# Interface principal
upload = st.file_uploader("Carregue imagem:", type=["jpg","png"])

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, channels="BGR", caption="Original", use_column_width=True)
        
    with col2:
        resultado = processar_imagem(img, threshold, min_area_cm, blur_size)
        
        if resultado:
            img_processada, df_resultados, escala = resultado
            st.image(img_processada, channels="BGR", use_column_width=True)
            
            # Exibe tabela com métricas
            st.dataframe(df_resultados, hide_index=True)
            
            # Botão de download
            csv = df_resultados.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar resultados (CSV)",
                data=csv,
                file_name="areas_cerebros.csv",
                mime="text/csv"
            )
            
            st.metric("Escala", f"{escala:.4f} cm/pixel")
            st.metric("Total de Cérebros", len(df_resultados))
