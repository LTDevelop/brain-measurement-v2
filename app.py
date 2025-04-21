import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧠 Segmentação Avançada de Cérebros")

# Controles na barra lateral
with st.sidebar:
    st.header("🔧 Controles Avançados")
    
    # Controles de cor
    st.subheader("Faixa de Cores (HSV)")
    h_min = st.slider("Hue Mínimo", 0, 179, 0)
    h_max = st.slider("Hue Máximo", 0, 179, 179)
    s_min = st.slider("Saturation Mínima", 0, 255, 50)
    s_max = st.slider("Saturation Máxima", 0, 255, 255)
    v_min = st.slider("Value Mínimo", 0, 255, 150)  # Foca em tons claros
    v_max = st.slider("Value Máximo", 0, 255, 255)
    
    # Controles geométricos
    st.subheader("Filtros Geométricos")
    min_area_cm = st.slider("Área mínima (cm²)", 0.1, 5.0, 0.8, step=0.1)
    max_area_cm = st.slider("Área máxima (cm²)", 0.5, 20.0, 5.0, step=0.1)
    solidity_thresh = st.slider("Solidez Mínima", 0.7, 1.0, 0.85, help="Filtra formas irregulares")

def processar_imagem(img, params):
    # Converter para HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Criar máscara baseada nos parâmetros HSV
    lower = np.array([params['h_min'], params['s_min'], params['v_min']])
    upper = np.array([params['h_max'], params['s_max'], params['v_max']])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Operações morfológicas
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Detecção da régua (para cálculo de escala)
    roi = img[-150:, :, :]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bin_regua = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contornos_regua, _ = cv2.findContours(bin_regua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos_regua:
        st.error("⚠️ Régua não detectada! Posicione uma régua de 1cm na base.")
        return None
    
    # Calcular escala
    x_r, y_r, w_r, h_r = cv2.boundingRect(max(contornos_regua, key=cv2.contourArea))
    escala = 1.0 / w_r  # cm/pixel
    min_area_px = params['min_area_cm'] / (escala ** 2)
    max_area_px = params['max_area_cm'] / (escala ** 2)
    
    # Encontrar contornos na máscara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Processar cada contorno
    resultados = []
    img_resultado = img.copy()
    
    for i, cnt in enumerate(contornos):
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        if (min_area_px <= area <= max_area_px) and (solidity >= params['solidity_thresh']):
            x,y,w,h = cv2.boundingRect(cnt)
            
            # Desenhar e anotar
            cv2.drawContours(img_resultado, [cnt], -1, (0,255,0), 2)
            cv2.putText(img_resultado, f"{i+1}", (x,y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            resultados.append({
                "Cérebro": i+1,
                "Área (cm²)": round(area * (escala ** 2), 2),
                "Solidez": round(solidity, 2),
                "Centroide (x,y)": (x+w//2, y+h//2)
            })
    
    # Visualizações intermediárias para debug
    debug_imgs = {
        "Máscara HSV": mask,
        "Contornos": cv2.drawContours(np.zeros_like(img), contornos, -1, (0,255,0), 1),
        "Régua Detectada": cv2.rectangle(roi.copy(), (x_r,y_r), (x_r+w_r,y_r+h_r), (255,0,0), 2)
    }
    
    return img_resultado, debug_imgs, pd.DataFrame(resultados), escala

# Interface principal
upload = st.file_uploader("Carregue imagem:", type=["jpg","png","tif"])

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Coletar parâmetros
    params = {
        'h_min': h_min, 'h_max': h_max,
        's_min': s_min, 's_max': s_max,
        'v_min': v_min, 'v_max': v_max,
        'min_area_cm': min_area_cm,
        'max_area_cm': max_area_cm,
        'solidity_thresh': solidity_thresh
    }
    
    # Processamento
    resultado = processar_imagem(img, params)
    
    if resultado:
        img_final, debug_imgs, df_resultados, escala = resultado
        
        # Layout de resultados
        tab1, tab2 = st.tabs(["Resultado Final", "Debug"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, channels="BGR", caption="Original", use_column_width=True)
            with col2:
                st.image(img_final, channels="BGR", caption="Cérebros Detectados", use_column_width=True)
                
            st.dataframe(df_resultados, hide_index=True)
            
            # Botão de download
            csv = df_resultados.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Exportar Resultados (CSV)",
                data=csv,
                file_name="resultados_cerebros.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.warning("Ajuste os parâmetros na barra lateral com base nestas visualizações")
            for name, debug_img in debug_imgs.items():
                st.image(debug_img, caption=name, use_column_width=True)

# Guia de uso
with st.expander("🎚️ Como Ajustar os Parâmetros"):
    st.markdown("""
    **Para cérebros claros em fundo escuro:**
    - ✅ **Value Mínimo**: 150-200 (filtra fundos escuros)
    - ✅ **Saturation Mínima**: 50+ (remove tons cinza)
    
    **Para melhor precisão:**
    - 🔍 Aumente a **Solidez** para filtrar formas irregulares
    - 📏 Defina **Área Mín/Máx** conforme o tamanho esperado
    - 🖌️ Use a aba **Debug** para visualizar a máscara HSV
    """)
