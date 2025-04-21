import streamlit as st
import cv2
import numpy as np
import pandas as pd  # Adicionado para gerar a planilha

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
    
    # DETECÇÃO DE CÉREBROS CLAROS
    _, bin_cerebros = cv2.threshold(cinza, threshold, 255, cv2.THRESH_BINARY)
    
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
        
        if area > min_area and 0.3 < aspect_ratio < 3.0:
            cerebros.append(cnt)
    
    # Detecção da régua (parte inferior)
    roi = cinza[-100:, :]
    _, bin_regua = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
    contornos_regua, _ = cv2.findContours(bin_regua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcular escala se régua for detectada
    escala = None
    if contornos_regua:
        x_r, y_r, w_r, h_r = cv2.boundingRect(max(contornos_regua, key=cv2.contourArea))
        escala = 1.0 / w_r  # cm/pixel
    
    # Preparar resultados para a planilha (NOVO)
    resultados = []
    img_resultado = img.copy()
    
    for i, cnt in enumerate(cerebros):
        area_px = cv2.contourArea(cnt)
        area_cm = area_px * (escala ** 2) if escala else area_px
        
        # Desenhar contorno
        cv2.drawContours(img_resultado, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(img_resultado, f"{i+1}", (x,y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        # Adicionar dados para a planilha
        resultados.append({
            "ID": i+1,
            "Área (px²)": int(area_px),
            "Área (cm²)": round(area_cm, 4) if escala else "N/A",
            "Centro X": x + w//2,
            "Centro Y": y + h//2
        })
    
    # Desenhar régua se detectada
    if contornos_regua:
        cv2.rectangle(img_resultado, (x_r, y_r+img.shape[0]-100), 
                     (x_r+w_r, y_r+h_r+img.shape[0]-100), (255,0,0), 2)
    
    return img_resultado, len(cerebros), escala, pd.DataFrame(resultados)  # Retorna o DataFrame

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
            img_processada, num_cerebros, escala, df_resultados = resultado
            st.image(img_processada, channels="BGR", caption=f"Cérebros Detectados: {num_cerebros}", use_column_width=True)
            
            # Mostrar métricas
            st.metric("Total de Cérebros", num_cerebros)
            if escala:
                st.metric("Escala", f"{escala:.4f} cm/pixel")
            else:
                st.warning("Régua não detectada - medidas em pixels")
            
            # Mostrar e baixar planilha (NOVO)
            st.dataframe(df_resultados, hide_index=True)
            
            csv = df_resultados.to_csv(index=False, sep=";").encode('utf-8')
            st.download_button(
                label="📥 Baixar Planilha (CSV)",
                data=csv,
                file_name="areas_cerebros.csv",
                mime="text/csv"
            )

# Dicas de uso
with st.expander("💡 Dicas para melhor detecção"):
    st.markdown("""
    1. **Iluminação uniforme**: Evite sombras e reflexos
    2. **Fundo escuro**: Cérebros devem ser mais claros que o fundo
    3. **Régua preta**: Posicione na parte inferior
    4. **Ajuste fino**:
       - Aumente o **limiar** para capturar apenas áreas muito claras
       - Aumente a **área mínima** para ignorar fragmentos
    """)
