"""Décodage des images envoyées par les utilisateurs.

Les photos de téléphone portent souvent leur orientation dans l'EXIF, et les
PNG peuvent être transparents : sans traitement, l'article serait couché ou
posé sur un fond noir, alors que le catalogue est photographié droit sur fond
blanc. L'image est donc redressée puis aplatie sur du blanc.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Fichier illisible, corrompu ou hors limites."""


def decode_image(data: bytes, max_pixels: int = 40_000_000) -> Image.Image:
    """Image RGB redressée (EXIF) et aplatie sur fond blanc."""
    if not data:
        raise InvalidImageError("Fichier vide")
    try:
        with Image.open(io.BytesIO(data)) as probe:  # lit l'en-tête seulement
            width, height = probe.size
            if width * height > max_pixels:
                raise InvalidImageError(
                    f"Image trop grande ({width}×{height} px, maximum {max_pixels:,} pixels)"
                )
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise InvalidImageError("Format d'image non reconnu ou fichier corrompu") from e

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")
