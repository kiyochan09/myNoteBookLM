import json
from pathlib import Path


INPUT_PATH = Path("ocr_result.json")
OUTPUT_PATH = Path("reading_order.json")


def center(box):
    """box = [x1, y1, x2, y2] の中心座標"""
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def size(box):
    """幅・高さ"""
    x1, y1, x2, y2 = box

    return (
        max(1, x2 - x1),
        max(1, y2 - y1)
    )


def detect_orientation(box):
    """
    OCRブロックの形状から
    縦書き / 横書きを推定する。
    """

    width, height = size(box)

    if height > width * 1.5:
        return "vertical"

    if width > height * 1.5:
        return "horizontal"

    return "unknown"


def load_ocr_result():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def make_blocks(data):
    blocks = []

    for block in data.get("blocks", []):

        box = block.get("box")

        if not box or len(box) != 4:
            continue

        orientation = detect_orientation(box)

        cx, cy = center(box)

        blocks.append({
            "id": block["id"],
            "text": block["text"],
            "score": block.get("score"),
            "box": box,
            "orientation": orientation,
            "center_x": cx,
            "center_y": cy
        })

    return blocks


def vertical_order(blocks):
    """
    縦書きの場合。

    日本語の一般的な縦書き文書では、

        右の列
        ↓
        左の列

    の順に読みます。

    同じ列に複数ブロックがある場合は
    上から下へ並べます。
    """

    # まずX座標で列を作るための簡易グループ化
    columns = []

    for block in sorted(
        blocks,
        key=lambda b: b["center_x"],
        reverse=True
    ):

        placed = False

        for column in columns:

            representative_x = column["center_x"]

            # 列間距離の許容値
            if abs(block["center_x"] - representative_x) < 60:

                column["blocks"].append(block)

                # 平均Xを更新
                xs = [
                    b["center_x"]
                    for b in column["blocks"]
                ]

                column["center_x"] = sum(xs) / len(xs)

                placed = True
                break

        if not placed:

            columns.append({
                "center_x": block["center_x"],
                "blocks": [block]
            })

    # 右列 → 左列
    columns.sort(
        key=lambda c: c["center_x"],
        reverse=True
    )

    result = []

    for column in columns:

        # 同じ縦列は上 → 下
        column["blocks"].sort(
            key=lambda b: b["center_y"]
        )

        result.extend(column["blocks"])

    return result


def horizontal_order(blocks):
    """
    横書きの場合。

    基本的に

        上の行
        ↓
        下の行

    同じ行では

        左 → 右

    とします。
    """

    rows = []

    for block in sorted(
        blocks,
        key=lambda b: b["center_y"]
    ):

        placed = False

        _, height = size(block["box"])

        tolerance = max(
            20,
            height * 0.7
        )

        for row in rows:

            if abs(
                block["center_y"] -
                row["center_y"]
            ) < tolerance:

                row["blocks"].append(block)

                ys = [
                    b["center_y"]
                    for b in row["blocks"]
                ]

                row["center_y"] = sum(ys) / len(ys)

                placed = True
                break

        if not placed:

            rows.append({
                "center_y": block["center_y"],
                "blocks": [block]
            })

    rows.sort(
        key=lambda r: r["center_y"]
    )

    result = []

    for row in rows:

        row["blocks"].sort(
            key=lambda b: b["center_x"]
        )

        result.extend(row["blocks"])

    return result


def make_reading_order(blocks):
    """
    ページ全体の読み順を決める。
    """

    vertical = [
        b for b in blocks
        if b["orientation"] == "vertical"
    ]

    horizontal = [
        b for b in blocks
        if b["orientation"] == "horizontal"
    ]

    unknown = [
        b for b in blocks
        if b["orientation"] == "unknown"
    ]

    result = []

    # 現段階では、ページ全体で多数派を判定
    if len(vertical) > len(horizontal):

        print("ページ方向: 縦書き")

        result.extend(
            vertical_order(vertical)
        )

        # 判定できなかったものは
        # 座標順で後ろに追加
        unknown.sort(
            key=lambda b: (
                -b["center_x"],
                b["center_y"]
            )
        )

        result.extend(unknown)

        # 横書き要素があれば最後に追加
        result.extend(
            horizontal_order(horizontal)
        )

    else:

        print("ページ方向: 横書き")

        result.extend(
            horizontal_order(horizontal)
        )

        unknown.sort(
            key=lambda b: (
                b["center_y"],
                b["center_x"]
            )
        )

        result.extend(unknown)

        result.extend(
            vertical_order(vertical)
        )

    return result


def main():

    print("① OCR結果を読み込みます")

    data = load_ocr_result()

    blocks = make_blocks(data)

    print(f"② OCRブロック数: {len(blocks)}")

    if not blocks:
        print("OCRブロックがありません。")
        return

    print("③ 文字方向を判定します")

    for block in blocks:

        print(
            f"ID={block['id']:3} "
            f"{block['orientation']:10} "
            f"{block['text'][:40]}"
        )

    print("④ 読み順を決定します")

    ordered = make_reading_order(blocks)

    output = {
        "page": data.get("page", 1),
        "image": data.get("image"),
        "reading_order": []
    }

    for position, block in enumerate(ordered, start=1):

        output["reading_order"].append({
            "position": position,
            "id": block["id"],
            "orientation": block["orientation"],
            "text": block["text"],
            "score": block["score"],
            "box": block["box"]
        })

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

    print("⑤ 完了")
    print(
        f"⑥ 保存先: {OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()