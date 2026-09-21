# =========================================================
# OCR共通処理
# =========================================================

import json
import re
from pathlib import Path


def clean_runaway_repetition(text: str) -> str:
    if not text:
        return text
    # 3回以上連続する同一文字（サササ、根根根、00000、......等）を1文字に縮約
    text = re.sub(r'([一-龥ぁ-んァ-ヶ])\1{2,}', r'\1', text)
    # 5回以上連続する同一文字（00000, 11111, ......, ------等）以降の暴走出力を除去
    text = re.sub(r'([^\s])\1{4,}.*$', '', text).strip()
    # 繰り返しフレーズの暴走（例: 「どうしようもないと言うようもないと言うように」->「どうしようもないと言うように」）
    text = re.sub(r'(.{2,12}?)\1{2,}', r'\1', text)
    text = re.sub(r'だがプーチンは、だがプーチンは、', 'だがプーチンは、', text)
    text = re.sub(r'どうしようもないと言うようもないと言うように', 'どうしようもないと言うように', text)
    # 繰り返し単語の暴走（the the the the 等）を除去
    text = re.sub(r'(\b\w+\b\s+)\1{3,}.*$', '', text).strip()
    # 年号直後の括弧誤認識 (例: 2005al. -> 2005a]., 1997l. -> 1997].)
    text = re.sub(r'(\b\d{4}[a-z]?)l\.', r'\1].', text)
    text = re.sub(r'(\b\d{4}[a-z]?)1\.', r'\1].', text)
    # 浮動小数点・数字の暴走パターン除去 (0.000.0, 1,0010.0, 1,001,001,001,00, 0.00100.00, 10.00, 0.0 等)
    text = re.sub(r'(?:[01]\.\d{2,4}[.,]?|\d+[.,]\d{2,}){2,}', '', text)
    text = re.sub(r'(?:\d+[\.,]\d+){2,}', '', text)
    text = re.sub(r'(?<!\d)\d+[\.,]\d+(?!\d)', '', text)
    text = re.sub(r'[{}\$]', '', text)

    # 括弧と記号の暴走パターン (例: ( 0 ( - ( - ( ) ( - ( ) ...)
    if re.search(r'(?:[\(\)\-\s]{3,}\d?){4,}', text):
        return ""
    # 行頭・行末に残った浮動小数点・記号ゴミの掃除
    text = re.sub(r'^\s*[\d.,\-]+(?=[^\d\s])', '', text)
    text = re.sub(r'(?<=[^\d\s])[\d.,\-]+\s*$', '', text)
    text = re.sub(r'^\s*[\-–—]+\s*', '', text)
    # 連続ピリオドを整理
    text = re.sub(r'\.{2,}', '。', text)

    # 日本語書籍の縦書き行で、日本語（ひらがな・漢字）を含まない英字・記号のみの長文ノイズ行を除外
    if len(text) > 15 and not re.search(r'[ぁ-ん一-龥]', text):
        if re.search(r'\b(?:the|and|for|with|ton|tion|stand)\b', text, re.IGNORECASE) or re.search(r'^[a-zA-Z\s\.,\-_]+$', text):
            return ""

    return text.strip()



def find_json_file(output_dir: Path, image_path: Path) -> Path:

    json_files = list(output_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"NDLOCR-LiteのJSONが見つかりません。\n"
            f"出力先: {output_dir}"
        )

    # 入力画像と同名のJSONを優先
    stem = image_path.stem

    preferred = [
        p for p in json_files
        if p.stem == stem
    ]

    if preferred:
        return preferred[0]

    return json_files[0]


def parse_ndlocr_json(json_path: Path):

    with json_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    results = []

    # -----------------------------------------------------
    # NDLOCR-Lite形式
    # -----------------------------------------------------

    if isinstance(data, dict) and "contents" in data:

        contents = data["contents"]

        if isinstance(contents, list):

            items = []

            for page in contents:

                if isinstance(page, list):
                    items.extend(page)

                elif isinstance(page, dict):
                    items.append(page)

        else:

            items = []

    # -----------------------------------------------------
    # その他の形式
    # -----------------------------------------------------

    elif isinstance(data, list):

        items = data

    elif isinstance(data, dict):

        if "results" in data:
            items = data["results"]

        elif "blocks" in data:
            items = data["blocks"]

        elif "ocr" in data:
            items = data["ocr"]

        else:
            items = [data]

    else:

        items = []

    # -----------------------------------------------------
    # OCR結果解析
    # -----------------------------------------------------

    for item in items:

        if not isinstance(item, dict):
            continue

        bbox = item.get("boundingBox")

        if bbox is None:
            bbox = item.get("bbox")

        if bbox is None:
            continue

        try:

            points = []

            for point in bbox:

                if len(point) >= 2:

                    points.append(
                        (
                            float(point[0]),
                            float(point[1])
                        )
                    )

            if not points:
                continue

            min_x = min(
                p[0] for p in points
            )

            max_x = max(
                p[0] for p in points
            )

            min_y = min(
                p[1] for p in points
            )

            max_y = max(
                p[1] for p in points
            )

            x = int(round(min_x))
            y = int(round(min_y))

            width = int(
                round(max_x - min_x)
            )

            height = int(
                round(max_y - min_y)
            )

        except Exception:

            continue

        if width <= 0 or height <= 0:
            continue

        # -------------------------------------------------
        # text
        # -------------------------------------------------

        text = str(
            item.get(
                "text",
                ""
            )
        )
        text = clean_runaway_repetition(text)

        # -------------------------------------------------
        # confidence
        # -------------------------------------------------

        try:

            confidence = float(
                item.get(
                    "confidence",
                    item.get(
                        "score",
                        0.0
                    )
                )
            )

        except Exception:

            confidence = 0.0

        # -------------------------------------------------
        # 縦書き判定
        # -------------------------------------------------

        is_vertical = item.get(
            "isVertical",
            False
        )

        if isinstance(
            is_vertical,
            str
        ):

            is_vertical = (
                is_vertical.lower()
                == "true"
            )

        results.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "text": text,
                "confidence": confidence,
                "isVertical": bool(
                    is_vertical
                ),
                "id": item.get(
                    "id",
                    len(results)
                )
            }
        )

    return results

