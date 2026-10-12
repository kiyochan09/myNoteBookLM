"""Keep media extension truthful and convert browser-unsupported image formats."""
import io
from PIL import Image


def browser_image(data, extension):
    for signature, suffix in [(b'\x89PNG\r\n\x1a\n','.png'),(b'\xff\xd8\xff','.jpg'),(b'GIF87a','.gif'),(b'GIF89a','.gif'),(b'BM','.bmp')]:
        if data.startswith(signature): return data, suffix
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP': return data, '.webp'
    if b'<svg' in data[:2048]:
        return data, '.svg'
    with Image.open(io.BytesIO(data)) as image:
        kind = image.format
        supported = {'PNG':'.png','JPEG':'.jpg','GIF':'.gif','WEBP':'.webp','BMP':'.bmp','AVIF':'.avif'}
        if kind in supported:
            return data, supported[kind]
        if kind == 'WMF': image.load(dpi=144)
        else: image.load()
        if image.mode not in ('RGB','RGBA','L','LA','P','1'):
            image = image.convert('RGB')
        output = io.BytesIO();image.save(output,format='PNG')
        return output.getvalue(), '.png'
