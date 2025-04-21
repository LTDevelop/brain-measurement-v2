import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧠 Medidor de Cérebros - Versão Final")

# Barra lateral com controles
with st.sidebar:
    st.header("⚙️ Controles de Detecção")
    threshold = st.slider("Limiar de brilho", 0, 255, 180, help="Valores mais altos = áreas mais claras")
    min_area = st.slider("Área mínima (pixels)", 50, 500, 150)
    blur_size = st.slider("Suavização", 1, 15, 5, step=2)

def processar_imagem(img, threshold, min_area, blur_size):
    # Pré-processamento
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (blur_size, blur_size), 0)
    
    # Segmentação dos cérebros (áreas claras)
    _, bin_cerebros = cv2.threshold(cinza, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    bin_cerebros = cv2.morphologyEx(bin_cerebros, cv2.MORPH_CLOSE, kernel)
    
    # Detecção dos cérebros
    contornos, _ = cv2.findContours(bin_cerebros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cerebros = [cnt for cnt in contornos if cv2.contourArea(cnt) > min_area]

    ####################################################
    # DETECÇÃO MELHORADA DA RÉGUA (PARTE CRÍTICA)
    ####################################################
    roi_regua = cinza[-150:, :]  # Região de interesse (parte inferior)
    _, bin_regua = cv2.threshold(roi_regua, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detecta linhas horizontais (traços da régua)
    lines = cv2.HoughLinesP(bin_regua, 1, np.pi/180, threshold=50, 
                          minLineLength=50, maxLineGap=10)
    
    escala = None
    if lines is not None:
        # Filtra apenas linhas horizontais (ângulo < 5 graus)
        linhas_horizontais = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if angle < 5:
                linhas_horizontais.append(line)
        
        if len(linhas_horizontais) >= 2:
            # Calcula a distância entre o primeiro e último traço
            x_coords = [x for line in linhas_horizontais for x in [line[0][0], line[0][2]]]
            x_min, x_max = min(x_coords), max(x_coords)
            dist_total_pixels = x_max - x_min
            
            # Supondo que a régua visível tem 10cm
            escala = 10.0 / dist_total_pixels  # cm/pixel
            
            # Debug visual (opcional)
            cv2.rectangle(img, (x_min, img.shape[0]-150), 
                         (x_max, img.shape[0]), (255,0,255), 2)
    ####################################################

    # Preparar resultados
    img_resultado = img.copy()
    resultados = []
    
    for i, cnt in enumerate(cerebros):
        area_px = cv2.contourArea(cnt)
        area_cm = area_px * (escala ** 2) if escala else 0
        
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.drawContours(img_resultado, [cnt], -1, (0,255,0), 2)
        cv2.putText(img_resultado, f"{i+1}", (x,y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        resultados.append({
            "ID": i+1,
            "Área (cm²)": round(area_cm, 2),
            "Centroide X": x + w//2,
            "Centroide Y": y + h//2
        })

    return img_resultado, len(cerebros), escala, pd.DataFrame(resultados)

# Interface principal
upload = st.file_uploader("Carregue imagem com cérebros e régua de 10cm na base:", 
                         type=["jpg", "png", "tif"])

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
            st.image(img_processada, channels="BGR", 
                   caption=f"Cérebros Detectados: {num_cerebros}", 
                   use_column_width=True)
            
            if escala:
                st.success(f"Escala calculada: 1cm = {1/escala:.1f} pixels")
                st.metric("Área Média", f"{df_resultados['Área (cm²)'].mean():.2f} cm²")
            else:
                st.error("Régua não detectada! Verifique se está visível na base.")
            
            st.dataframe(df_resultados, hide_index=True)
            
            # Botão de download
            csv = df_resultados.to_csv(index=False, sep=";").encode('utf-8')
            st.download_button(
                "📥 Exportar Resultados",
                data=csv,
                file_name="areas_cerebros.csv",
                mime="text/csv"
            )

# Dicas de uso
with st.expander("💡 Instruções de uso"):
    st.markdown("""
    1. **Posicione a régua**: Deve estar na **parte inferior** da imagem, com os traços de 1 a 10cm visíveis
    2. **Iluminação**: Fundo escuro e cérebros claros funcionam melhor
    3. **Ajustes finos**:
       - Aumente o **limiar** se detectar muito fundo
       - Aumente a **área mínima** para filtrar fragmentos
    4. **Verifique a régua**: A área rosa na imagem processada deve cobrir toda a régua
    """)
