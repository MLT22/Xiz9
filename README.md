# Xiz9
APi para a verificar se uma imagem foi ou não gerada por ia

---
## Como rodar:

    1.No cmd ou powershell rode:
    
        `python -m venv .venv`
        
    2.Ativar o ambiente virtual:
    
        No PowerShell:
        `.venv\Scripts\Activate.ps1`
        
        No cmd:
        `.venv\Scripts\activate.bat`
        
    3.Verifique se está no ambiente virtual python(venv):
    
        No terminal vai aparecer (.venv)
        
    4.Instalar as dependencias:
    
        `pip install -r requirements.txt`

    5.Arrumar os modelos nas pastas:
        Baixar os modelos no drive e organizar da seguinte maneira:
        Xiz9/
        │
        ├── modelos/
        │   ├── PCA_anomaly/
        │   │   └── modelo_pca_v2.pkl   
        │   ├── Luminescencia/
        │   │   ├── scaler_luminance.pkl
        │   │   └── svm_luminance.pkl
        │   ├── Ruido
        │   │   └── modelo_ruido.pkl
        │   └── Stacking
        │       └── modelo_stacking.pkl   
        
        etc.
        
    6.Rodar a aplciação:
    
        Rode esse comando no cmd ou powershell:
        
        `fastapi dev`
        
    7.Testar:
    
        Abra o link da documentação que apareceu no terminal
