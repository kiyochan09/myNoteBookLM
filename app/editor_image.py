"""エディタへ保存する図版を1MB以下に整える。"""
from io import BytesIO
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 1_000_000

def prepare_editor_image(content):
    if not content:
        raise ValueError('画像ファイルが空です。')
    try:
        with Image.open(BytesIO(content)) as source:
            source.load()
            image=ImageOps.exif_transpose(source).copy()
            original_format=source.format
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise ValueError('読み取れる画像ファイルを選択してください。') from exc
    # 小さい標準画像は画質を維持する。向き情報があるものは正規化する。
    if len(content)<=MAX_IMAGE_BYTES and original_format in ('PNG','JPEG'):
        with Image.open(BytesIO(content)) as source:
            if not source.getexif().get(274):
                return content,('.png' if original_format=='PNG' else '.jpg'),image.width,image.height
    image.thumbnail((4096,4096),Image.Resampling.LANCZOS)
    for _ in range(20):
        buf=BytesIO();image.save(buf,format='PNG',optimize=True)
        if buf.tell()<=MAX_IMAGE_BYTES:return buf.getvalue(),'.png',image.width,image.height
        # 写真はJPEGで圧縮。透明な図版はPNGのまま縮小する。
        transparent=image.mode in ('RGBA','LA') or 'transparency' in image.info
        if not transparent:
            for quality in (92,85,75,65):
                buf=BytesIO();image.convert('RGB').save(buf,format='JPEG',quality=quality,optimize=True)
                if buf.tell()<=MAX_IMAGE_BYTES:return buf.getvalue(),'.jpg',image.width,image.height
        image=image.resize((max(1,int(image.width*.8)),max(1,int(image.height*.8))),Image.Resampling.LANCZOS)
    raise ValueError('画像を1MB以下に調整できませんでした。小さい画像を選択してください。')
