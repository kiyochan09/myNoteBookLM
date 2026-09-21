import json
from pathlib import Path


INPUT_PATH = Path("ocr_result.json")
OUTPUT_PATH = Path("classified_result.json")


def load_result():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_block(block, page_width, page_height):
    text = block.get("text", "").strip()
    box = block.get("box")

    if not box or len(box) != 4:
        return "unknown"

    x1, y1, x2, y2 = box

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # ---------------------------------
    # 1. ページ番号候補
    # ---------------------------------
    if len(text) <= 6:
        cleaned = text.replace(" ", "").replace("　", "")

        if cleaned.isdigit():
            return "page_number"

    # ---------------------------------
    # 2. 上端・下端の候補
    # ---------------------------------
    if page_height:
        top_ratio = y1 / page_height
        bottom_ratio = y2 / page_height

        if top_ratio < 0.08:
            return "header_candidate"

        if bottom_ratio > 0.92:
            return "footer_candidate"

    # ---------------------------------
    # 3. 縦書き本文
    # ---------------------------------
    if height > width * 1.5:
        return "body_vertical_candidate"

    # ---------------------------------
    # 4. 横書き
    # ---------------------------------
    if width > height * 1.5:
        return "horizontal_candidate"

    # ---------------------------------
    # 5. その他
    # ---------------------------------
    return "unknown"


def main():

    print("① OCR結果を読み込みます")

    data = load_result()

    blocks = data.get("blocks", [])

    print(f"② ブロック数: {len(blocks)}")

    # ---------------------------------
    # ページサイズを推定
    # ---------------------------------
    max_x = 0
    max_y = 0

    for block in blocks:

        box = block.get("box")

        if not box or len(box) != 4:
            continue

        max_x = max(max_x, box[2])
        max_y = max(max_y, box[3])

    page_width = data.get("width") or max_x
    page_height = data.get("height") or max_y

    print(f"③ 推定ページサイズ: {page_width} x {page_height}")

    # ---------------------------------
    # 分類
    # ---------------------------------
    classified_blocks = []

    for block in blocks:

        category = classify_block(
            block,
            page_width,
            page_height
        )

        new_block = dict(block)

        new_block["category"] = category

        classified_blocks.append(new_block)

        print(
            f"ID={block['id']:3} "
            f"{category:25} "
            f"{block['text'][:50]}"
        )

    # ---------------------------------
    # 保存
    # ---------------------------------
    output = {
        "page": data.get("page", 1),
        "image": data.get("image"),
        "width": page_width,
        "height": page_height,
        "blocks": classified_blocks
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("④ 分類完了")
    print(f"⑤ 保存先: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()