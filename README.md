# Xiz9
APi para a verificar se uma imagem foi ou não gerada por ia

---
## Como rodar:

    1.No cmd ou powershell rode:
    
        `python -m venv .venv`
        
    2.Verifique se está no ambiente virtual python(venv):
    
        No cmd vai aparece (.venv)
        
    3.Instalar as dependencias:
    
        `pip install -r requirements.txt`

    4.Arrumar os modelos nas pastas:
        Baixar os modelos no drive e organizar da seguinte maneira:
        Xiz9/
        │
        ├── modelos/
        │   ├── PCA_anomaly/
        │   │   └── modelo_pca_v2.pkl   
        │   ├── Luminescencia/
        │   │   └── scaler_luminance.pkl
        │   │   └── svm_luminance.pkl
        │   ├── Ruido
        │   │   └── modelo_ruido.pkl
        │   └── Stacking
        │       └── modelo_stacking.pkl   
        
        etc.
        
    4.Rodar a aplciação:
    
        Rode esse comando no cmd ou powershell:
        
        `fastapi dev`
        
    5.Testar:
    
        Abra o link da documentação que apareceu no terminal
