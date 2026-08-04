import io
import json

import piexif
from PIL import Image, PngImagePlugin

_AI_KEYWORDS = [
    "stable diffusion", "stablediffusion", "automatic1111", "a1111",
    "comfyui", "comfy ui", "novelai", "invokeai", "midjourney",
    "dall-e", "dalle", "firefly", "adobe firefly", "imagen",
    "dreamstudio", "openjourney", "kandinsky", "chatgpt"
]

_AI_PNG_KEYS = {"parameters", "workflow", "prompt", "invokeai_metadata", "sd-metadata"}


class MetadataService:

    @staticmethod
    def extract(filename: str, content_type: str, contents: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise ValueError("Não foi possível processar o arquivo como imagem.")

        width, height = image.size
        return {
            "filename": filename,
            "content_type": content_type,
            "file_size_bytes": len(contents),
            "format": image.format,
            "mode": image.mode,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 4) if height else None,
            "is_animated": getattr(image, "is_animated", False),
            "frames": getattr(image, "n_frames", 1),
            "exif": MetadataService._all_metadata(image, contents),
        }

    @staticmethod
    def _all_metadata(image: Image.Image, contents: bytes) -> dict:
        all_metadata = {}

        if isinstance(image, PngImagePlugin.PngImageFile):
            png_info = image.info or {}
            for key, value in png_info.items():
                if isinstance(value, bytes):
                    try:
                        all_metadata[key] = value.decode("utf-8", errors="replace")
                    except Exception:
                        all_metadata[key] = repr(value)
                elif isinstance(value, str):
                    try:
                        all_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        all_metadata[key] = value
                else:
                    all_metadata[key] = value

        try:
            exif_bytes = image.info.get("exif")
            if exif_bytes:
                exif_dict = piexif.load(exif_bytes)
                for ifd_name, ifd_data in exif_dict.items():
                    if not isinstance(ifd_data, dict):
                        continue
                    for tag_id, value in ifd_data.items():
                        tag_name = piexif.TAGS[ifd_name].get(tag_id, {}).get("name", str(tag_id))
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace").strip("\x00")
                        all_metadata.setdefault("exif_piexif", {})[tag_name] = value
        except Exception:
            pass

        xmp = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")
        if xmp:
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", errors="replace")
            all_metadata["xmp"] = xmp

        return all_metadata

    @staticmethod
    def check_ai_indicators(image: Image.Image, raw_metadata: dict) -> dict:
        indicators = []

        # Chaves PNG exclusivas de geradores de IA
        if isinstance(image, PngImagePlugin.PngImageFile):
            found_keys = _AI_PNG_KEYS & set(image.info.keys())
            for k in found_keys:
                indicators.append(f"PNG key '{k}' encontrada (comum em SD/ComfyUI)")

        # Varredura por keywords em todos os valores de metadados
        def _search(obj, path=""):
            if isinstance(obj, str):
                low = obj.lower()
                for kw in _AI_KEYWORDS:
                    if kw in low:
                        indicators.append(f"Keyword '{kw}' em {path}")
                        return
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _search(v, path=f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _search(v, path=f"{path}[{i}]")

        _search(raw_metadata)

        return {
            "has_ai_indicators": len(indicators) > 0,
            "indicators": indicators,
        }

    @staticmethod
    def check(contents: bytes) -> dict:
        """Wrapper seguro para uso pelos orquestradores (BronzeService):
        nunca levanta exceção, cai para 'sem indícios' se a imagem não puder ser lida."""
        try:
            image = Image.open(io.BytesIO(contents))
            raw_metadata = MetadataService._all_metadata(image, contents)
            return MetadataService.check_ai_indicators(image, raw_metadata)
        except Exception:
            return {"has_ai_indicators": False, "indicators": []}
