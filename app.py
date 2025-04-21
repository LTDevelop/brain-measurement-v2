import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import pandas as pd
from ipywidgets import widgets, Layout
from IPython.display import display, clear_output
import os

class BrainMeasure:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.image = None
        self.label_image = None
        self.regions = None
        self.output = widgets.Output()
        self.download_button = widgets.Button(description="Baixar Imagem", layout=Layout(width='150px'))
        self.download_button.on_click(self.download_image)
        
        # Widgets para controle
        self.threshold_slider = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01, description='Threshold:')
        self.area_slider = widgets.IntSlider(value=100, min=0, max=1000, step=10, description='Área Mín:')
        self.process_button = widgets.Button(description="Processar Imagem")
        
        # Atualizações interativas
        self.threshold_slider.observe(self.update_plot, names='value')
        self.area_slider.observe(self.update_plot, names='value')
        self.process_button.on_click(self.process_image)
        
        # Widget de upload
        self.upload = widgets.FileUpload(accept='.jpg,.png,.tif', multiple=False)
        
        # Exibir widgets
        display(widgets.VBox([
            self.upload,
            widgets.HBox([self.threshold_slider, self.area_slider]),
            self.process_button,
            self.download_button
        ]))
        display(self.output)
        
    def process_image(self, change):
        if not self.upload.value:
            with self.output:
                print("Por favor, faça upload de uma imagem primeiro.")
            return
            
        # Carregar imagem
        upload_filename = next(iter(self.upload.value))
        image_data = self.upload.value[upload_filename]['content']
        nparr = np.frombuffer(image_data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Processar imagem
        self.update_plot()
        
    def update_plot(self, change=None):
        if self.image is None:
            return
            
        with self.output:
            clear_output(wait=True)
            
            # Converter para escala de cinza e aplicar threshold
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, int(self.threshold_slider.value*255), 255, cv2.THRESH_BINARY_INV)
            
            # Remover pequenos ruídos
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Rotular regiões
            self.label_image = measure.label(opening, connectivity=2)
            self.regions = measure.regionprops(self.label_image)
            
            # Filtrar regiões por área mínima
            min_area = self.area_slider.value
            valid_regions = [r for r in self.regions if r.area >= min_area]
            
            # Desenhar resultados
            self.ax.clear()
            self.ax.imshow(self.image)
            
            # Criar tabela de dados
            data = []
            
            for i, region in enumerate(valid_regions, 1):  # Começando de 1
                # Desenhar retângulo ao redor do objeto
                minr, minc, maxr, maxc = region.bbox
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
                self.ax.add_patch(rect)
                
                # Adicionar número de identificação
                self.ax.text(minc, minr, str(i), color='blue', fontsize=12, 
                            bbox=dict(facecolor='yellow', alpha=0.5))
                
                # Coletar dados
                data.append([i, region.area, region.eccentricity, region.solidity])
            
            self.ax.set_title(f"{len(valid_regions)} cérebros detectados")
            plt.tight_layout()
            
            # Mostrar tabela
            if data:
                df = pd.DataFrame(data, columns=['ID', 'Área', 'Excentricidade', 'Solidez'])
                display(df)
            
            plt.show()
    
    def download_image(self, b):
        if self.image is None or self.label_image is None:
            with self.output:
                print("Nenhuma imagem processada para baixar.")
            return
            
        # Criar diretório se não existir
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        
        # Salvar figura
        output_path = 'outputs/brain_measurement.png'
        self.fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        with self.output:
            clear_output(wait=True)
            print(f"Imagem salva em: {output_path}")

# Criar instância da classe
brain_measurer = BrainMeasure()
