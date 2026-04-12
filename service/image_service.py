import io
import json
import piexif
from PIL import Image, PngImagePlugin


class ImageService:

    @staticmethod
    def extract_metadata(filename: str, content_type: str, contents: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise ValueError("Não foi possível processar o arquivo como imagem.")

        width, height = image.size

        metadata = {
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
            "exif": ImageService._extract_all_metadata(image, contents),
        }

        return metadata

    @staticmethod
    def _extract_all_metadata(image: Image.Image, contents: bytes) -> dict:
        all_metadata = {}

        # Metadados PNG (onde IA costuma guardar prompt, seed, etc.)
        if isinstance(image, PngImagePlugin.PngImageFile):
            png_info = image.info or {}
            for key, value in png_info.items():
                if isinstance(value, bytes):
                    try:
                        all_metadata[key] = value.decode("utf-8", errors="replace")
                    except Exception:
                        all_metadata[key] = repr(value)
                elif isinstance(value, str):
                    # Tenta parsear como JSON (comum em ComfyUI)
                    try:
                        all_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        all_metadata[key] = value
                else:
                    all_metadata[key] = value

        # Metadados EXIF via piexif (JPEG/TIFF)
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

        # Metadados XMP
        xmp = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")
        if xmp:
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", errors="replace")
            all_metadata["xmp"] = xmp

        return all_metadata
    
    def __init__():
        return