# =========================================================
# 表見出し検出
# =========================================================


def extract_table_captions(
    body_results,
    table_regions
):
    """
    表の直上にあるOCRを表見出し候補として抽出する。

    表より下にあるOCRは対象外とし、表と横方向に
    一定以上重なっている、表直上の最も近いOCRを採用する。
    """

    captions = []

    caption_max_gap = 40

    for table_index, table in enumerate(
        table_regions,
        start=1
    ):
        tx1 = table["x"]
        ty1 = table["y"]
        tx2 = tx1 + table["width"]

        candidates = []

        for ocr in body_results:
            ox1 = ocr["x"]
            oy1 = ocr["y"]
            ox2 = ox1 + ocr["width"]
            oy2 = oy1 + ocr["height"]

            # 表より上にあるOCRだけを対象にする
            if oy2 > ty1:
                continue

            gap = ty1 - oy2

            if gap > caption_max_gap:
                continue

            # 表と横方向に重なっているか確認
            overlap_x1 = max(ox1, tx1)
            overlap_x2 = min(ox2, tx2)

            if overlap_x2 <= overlap_x1:
                continue

            overlap_width = overlap_x2 - overlap_x1
            ocr_width = max(1, ox2 - ox1)
            overlap_ratio = overlap_width / ocr_width

            if overlap_ratio < 0.30:
                continue

            candidates.append(ocr)

        if candidates:
            candidates.sort(
                key=lambda r: (
                    ty1 - (r["y"] + r["height"]),
                    r["x"]
                )
            )

            captions.append(
                {
                    "table_index": table_index,
                    "ocr": candidates[0]
                }
            )

    return captions

