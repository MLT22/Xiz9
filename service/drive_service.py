import io

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import GOOGLE_API_KEY, DRIVE_FOLDER_ID


class DriveService:
    """Lista e baixa imagens de uma pasta pública do Google Drive."""

    _client = None

    @staticmethod
    def _get_client():
        """Constroi o cliente do Drive uma unica vez e reaproveita."""
        if DriveService._client is None:
            DriveService._client = build(
                'drive', 'v3',
                developerKey=GOOGLE_API_KEY,
                cache_discovery=False,
            )
        return DriveService._client

    @staticmethod
    def listar_imagens():
        """Retorna as imagens da pasta, da mais recente para a mais antiga."""
        drive = DriveService._get_client()

        resultado = drive.files().list(
            q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false",
            fields='files(id, name, mimeType, createdTime)',
            orderBy='createdTime desc',
            pageSize=100,
        ).execute()

        return [
            {
                'id': f['id'],
                'nome': f['name'],
                'tipo': f['mimeType'],
                'criado_em': f.get('createdTime'),
            }
            for f in resultado.get('files', [])
        ]

    @staticmethod
    def baixar_imagem(file_id: str):
        """Baixa os bytes de uma imagem. Retorna (bytes, mime_type)."""
        drive = DriveService._get_client()

        meta = drive.files().get(fileId=file_id, fields='mimeType').execute()
        mime_type = meta.get('mimeType', 'application/octet-stream')

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, drive.files().get_media(fileId=file_id))
        concluido = False
        while not concluido:
            _, concluido = downloader.next_chunk()

        buffer.seek(0)
        return buffer.read(), mime_type