

DRIVE_FOLDER_ID = "1eCXWljjRKuutkrfH-6q4u1i7CXuLHJrzv84k9HIF76N0Js874zW00sK0CHRZPEApWbt4Fb-j"

try:
    from config_local import GOOGLE_API_KEY
except ImportError:
    GOOGLE_API_KEY = ""
    print("⚠️  config_local.py não encontrado — crie com GOOGLE_API_KEY = '...'")