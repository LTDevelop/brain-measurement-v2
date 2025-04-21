import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import pandas as pd
import os
from PIL import Image
import io

# Configuração da página
st.set_page_config(layout="wide", page_title="Medição de Cérebros")

class BrainMeasure:
    def __init__(self):
        self.image = None
        self.label_image = None
        self.regions = None
        self.fig = None
        
        # Adicionar campo para escala (pixels por cm)
        st.sidebar.header("Configurações de Escala")
        self.pixels_per_cm = st.sidebar.number_input(
            "Pixels por centímetro (escala):",
            min_value=1,
            value=100,  # Valor padrão - ajuste conforme sua calibração
            help="Quantos pixels equivalem a 1 cm na imagem"
        )
        
        # Widgets para controle
        st.sidebar.header("Parâmetros de Processamento")
        self.threshold = st.sidebar.slider(
            "Threshold:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01
        )
        
        self.min_area_cm2 = st.sidebar.slider(
            "Área Mínima (cm²):", 
            min_value=0.0, 
            max_value=10.0, 
            value=1.0, 
            step=0.1
        )
        
        # Upload de imagem
        st.header("Upload de Imagem")
        uploaded_file = st.file_uploader(
            "Carregue uma imagem de cérebros", 
            type=['jpg', 'jpeg', 'png', 'tif']
        )
        
        if uploaded_file is not None:
            self.process_image(uploaded_file)
    
    def pixels_to_cm2(self, area_pixels):
        """Converte área em pixels para cm²"""
        return area_pixels / (self.pixels_per_cm ** 2)
    
    def cm2_to_pixels(self, area_cm2):
        """Converte área em cm² para pixels"""
        return area_cm2 * (self.pixels_per_cm ** 2)
    
    def process_image(self, uploaded_file):
        # Carregar imagem
        image_data = uploaded_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Processar imagem
        self.update_plot()
        
    def update_plot(self):
        if self.image is None:
            return
            
        # Converter para escala de cinza e aplicar threshold
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, int(self.threshold*255), 255, cv2.THRESH_BINARY_INV)
        
        # Remover pequenos ruídos
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Rotular regiões
        self.label_image = measure.label(opening, connectivity=2)
        self.regions = measure.regionprops(self.label_image)
        
        # Filtrar regiões por área mínima (convertendo cm² para pixels)
        min_area_pixels = self.cm2_to_pixels(self.min_area_cm2)
        valid_regions = [r for r in self.regions if r.area >= min_area_pixels]
        
        # Criar figura
        self.fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.image)
        
        # Criar tabela de dados
        data = []
        
        for i, region in enumerate(valid_regions, 1):  # Começando de 1
            # Desenhar retângulo ao redor do objeto
            minr, minc, maxr, maxc = region.bbox
            rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # Adicionar número de identificação
            ax.text(minc, minr, str(i), color='blue', fontsize=12, 
                    bbox=dict(facecolor='yellow', alpha=0.5))
            
            # Coletar dados (convertendo área para cm²)
            area_cm2 = self.pixels_to_cm2(region.area)
            data.append([i, area_cm2, region.eccentricity, region.solidity])
        
        ax.set_title(f"{len(valid_regions)} cérebros detectados")
        plt.tight_layout()
        
        # Mostrar resultados no Streamlit
        st.pyplot(self.fig)
        
        # Mostrar tabela
        if data:
            df = pd.DataFrame(data, columns=['ID', 'Área (cm²)', 'Excentricidade', 'Solidez'])
            st.dataframe(df)
            
            # Botão para download
            self.download_image()
    
    def download_image(self):
        if self.fig is None:
            return
            
        # Salvar figura em um buffer
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Botão de download
        st.download_button(
            label="Baixar Imagem Anotada",
            data=buf,
            file_name="brain_measurement.png",
            mime="image/png"
        )

# Criar instância da classe
if __name__ == '__main__':
    brain_measurer = BrainMeasure()
