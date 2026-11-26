import io

import pytest
from PIL import Image

from app.images import InvalidImageError, decode_image


def encode(img: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (80, 60), "white")
    img.paste((255, 0, 0), (0, 0, 10, 60))  # bande rouge à gauche
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation : rotation de 90° nécessaire à l'affichage
    out = decode_image(encode(img, "JPEG", exif=exif, quality=95))
    assert out.size == (60, 80)
    r, g, b = out.getpixel((30, 2))  # la bande se retrouve en haut
    assert r > 200 and g < 60 and b < 60


def test_transparent_png_is_flattened_on_white():
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    img.paste((0, 0, 255, 255), (5, 5, 15, 15))
    out = decode_image(encode(img))
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((10, 10)) == (0, 0, 255)


def test_palette_and_grayscale_images_become_rgb():
    assert decode_image(encode(Image.new("L", (10, 10), 128))).getpixel((0, 0)) == (128, 128, 128)
    assert decode_image(encode(Image.new("P", (10, 10)))).mode == "RGB"


def test_webp_is_supported():
    assert decode_image(encode(Image.new("RGB", (12, 12), "green"), "WEBP")).size == (12, 12)


@pytest.mark.parametrize("data", [b"", b"not an image", encode(Image.new("RGB", (50, 50)))[:60]])
def test_unreadable_files_are_rejected(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_too_many_pixels_rejected_before_decoding():
    with pytest.raises(InvalidImageError, match="trop grande"):
        decode_image(encode(Image.new("RGB", (100, 100))), max_pixels=5_000)
