"""Image-only candidate and value selection for horizontal two-digit glyphs."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ImageDigitMatcher:
    def __init__(self, template_dir):
        templates = []
        labels = []
        fonts = Path('C:/Windows/Fonts')
        for name in ('msmincho.ttc', 'msgothic.ttc', 'meiryo.ttc', 'arial.ttf', 'yumin.ttf', 'times.ttf'):
            path = fonts / name
            if not path.exists():
                continue
            # Tiny printed digits keep size-dependent pixel/stroke patterns.
            # Upscaling a larger glyph to 24x24 cannot reproduce those patterns,
            # so render the small sizes before applying the same normalization.
            for size in (*range(12, 24), 24, 28, 32, 36, 40):
                try:
                    font = ImageFont.truetype(str(path), size)
                except OSError:
                    continue
                for number in range(100):
                    image = Image.new('L', (64, 64), 255)
                    draw = ImageDraw.Draw(image)
                    box = draw.textbbox((0, 0), f'{number:02d}', font=font)
                    draw.text(((64 - box[2] + box[0]) // 2 - box[0],
                               (64 - box[3] + box[1]) // 2 - box[1]),
                              f'{number:02d}', font=font, fill=0)
                    templates.append(self.vector(np.array(image)))
                    labels.append(number)
        for path in sorted(Path(template_dir).glob('*.png')):
            number = path.stem.split('_')[0]
            if len(number) != 2 or not number.isascii() or not number.isdigit():
                continue
            image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                vector = self.vector(image)
                if vector is not None:
                    templates.append(vector)
                    labels.append(int(number))
        self.matrix = np.asarray(templates, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int32)

    @staticmethod
    def vector(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        points = cv2.findNonZero(binary)
        if points is None:
            return None
        x, y, width, height = cv2.boundingRect(points)
        vector = cv2.resize(gray[y:y + height, x:x + width], (24, 24),
                            interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        vector -= vector.mean()
        return vector / max(float(np.linalg.norm(vector)), 1e-6)

    def candidates(self, image):
        h, w = image.shape[:2]
        if h < 30 or h < w * 2 or not 12 <= w <= 80 or not self.matrix.size:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bounds = np.flatnonzero(np.diff(np.r_[False, (binary > 0).any(axis=1), False]))
        geometry, vectors = [], []
        for a, b in zip(bounds[::2], bounds[1::2]):
            x, _, width, height = cv2.boundingRect(cv2.findNonZero(binary[a:b]))
            if not (8 <= height <= 32 and 8 <= width <= 45 and .55 <= width / height <= 2.5):
                continue
            geometry.append((int(a), int(b), x, width))
            vectors.append(self.vector(image[a:b, x:x + width]))
        if not vectors:
            return []
        similarities = np.asarray(vectors) @ self.matrix.T
        results = []
        for (a, b, x, width), scores in zip(geometry, similarities):
            best = np.full(100, -1., dtype=np.float32)
            np.maximum.at(best, self.labels, scores)
            order = np.argsort(best)[::-1]
            number, second = int(order[0]), int(order[1])
            # A candidate is still provisional: the runtime independently
            # confirms exactly this image-selected value before placing it.
            if best[number] < .70 or best[number] - best[second] < .015:
                continue
            results.append({'y1': a, 'y2': b, 'x': x, 'width': width,
                            'digits': f'{number:02d}', 'score': float(best[number]),
                            'runner_up_score': float(best[second])})
        return results
