import streamlit as st
import cv2
import numpy as np
import pandas as pd  # Adicionado para exportar CSV

# Configurações iniciais
st.set_page_config(layout="wide")
st.title("🧠 Medidor de Cérebros - Versão Estável")

# Função principal (igual à que você aprovou)
def processar_imagem(image):
    # 1. Pré-processamento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Detectar régua (parte inferior)
    roi_height = 100
    roi = image[image.shape[0]-roi_height:, :]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, ruler_bin = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(ruler_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        st.error("⚠️ Régua não detectada! Posicione uma régua de 1cm na parte inferior.")
        return None
    
    # 3. Calcular escala
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    scale_cm = 1.0 / w  # cm por pixel
    
    # 4. Segmentar cérebros
    _, brain_bin = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    brain_contours, _ = cv2.findContours(brain_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filtrar contornos pequenos
    min_area = (50 * (1/scale_cm)) ** 2  # 50px² em unidades de imagem
    brains = [cnt for cnt in brain_contours if cv2.contourArea(cnt) > min_area]
    
    # 6. Calcular métricas
    total_area_px = sum(cv2.contourArea(cnt) for cnt in brains)
    total_area_cm2 = total_area_px * (scale_cm ** 2)
    
    # 7. Visualização
    result_img = image.copy()
    cv2.drawContours(result_img, brains, -1, (0, 255, 0), 3)
    cv2.rectangle(result_img, (x, y+image.shape[0]-roi_height), 
                 (x+w, y+h+image.shape[0]-roi_height), (255, 0, 0), 2)
    
    # 8. Criar tabela de resultados (NOVO)
    results = []
    for i, cnt in enumerate(brains):
        area_px = cv2.contourArea(cnt)
        area_cm2 = area_px * (scale_cm ** 2)
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        
        results.append({
            "ID": i+1,
            "Área (cm²)": round(area_cm2, 2),
            "Centro X": x_cnt + w_cnt//2,
            "Centro Y": y_cnt + h_cnt//2
        })
    
    return result_img, total_area_cm2, len(brains), pd.DataFrame(results)  # Retorna o DataFrame

# Interface do usuário
uploaded_file = st.file_uploader("Upload de imagem com régua de 1cm:", type=["jpg", "png"])

if uploaded_file is not None:
    # Carregar imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Processar
    results = processar_imagem(image)
    
    if results:
        processed_img, total_area, brain_count, df_results = results
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, channels="BGR", use_column_width=True)
        
        with col2:
            st.metric("Área Total", f"{total_area:.2f} cm²")
            st.metric("Número de Cérebros", brain_count)
            
            # Mostrar tabela (NOVO)
            st.dataframe(df_results, hide_index=True)
            
            # Botão de download (NOVO)
            csv = df_results.to_csv(index=False, sep=";").encode('utf-8')
            st.download_button(
                label="📥 Baixar Planilha (CSV)",
                data=csv,
                file_name="areas_cerebros.csv",
                mime="text/csv"
            )

# Instruções de uso
with st.expander("ℹ️ Instruções"):
    st.markdown("""
    1. **Posicione uma régua de 1cm na parte inferior da imagem**
    2. **Certifique-se de que os cérebros estão claramente visíveis**
    3. **Ajuste o limite (threshold) na barra lateral se necessário**
    """)
