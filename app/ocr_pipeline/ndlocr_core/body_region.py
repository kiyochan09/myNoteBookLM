# =========================================================
# 本文領域生成
# =========================================================


def create_body_regions(
    body_results
):
    """
    表セルに割り当てられていないOCR結果から
    本文候補領域を生成する。

    縦書き・横書きの両方に対応する。

    注意：
    現段階では既存アルゴリズムをそのまま分離している。
    本文方向をユーザー設定から受け取る仕様変更は
    リファクタリング完了後に別工程で行う。
    """

    if not body_results:
        return []

    # -----------------------------------------------------
    # OCR結果を縦書き・横書きに分離
    # -----------------------------------------------------

    vertical_results = [
        r
        for r in body_results
        if r.get("isVertical", False)
    ]

    horizontal_results = [
        r
        for r in body_results
        if not r.get("isVertical", False)
    ]

    regions = []

    # -----------------------------------------------------
    # 縦書き本文
    # -----------------------------------------------------

    if vertical_results:
        vertical_results.sort(
            key=lambda r: r["x"]
        )

        groups = []

        current_group = []

        column_gap = 30

        for r in vertical_results:

            if not current_group:
                current_group.append(r)
                continue

            previous = current_group[-1]

            gap = abs(
                r["x"] - previous["x"]
            )

            if gap <= column_gap:
                current_group.append(r)

            else:
                groups.append(
                    current_group
                )

                current_group = [r]

        if current_group:
            groups.append(
                current_group
            )

        # -------------------------------------------------
        # 最大グループを本文候補とする
        # -------------------------------------------------

        if groups:

            groups.sort(
                key=lambda g: len(g),
                reverse=True
            )

            main_group = groups[0]

            min_x = min(
                r["x"]
                for r in main_group
            )

            min_y = min(
                r["y"]
                for r in main_group
            )

            max_x = max(
                r["x"] + r["width"]
                for r in main_group
            )

            max_y = max(
                r["y"] + r["height"]
                for r in main_group
            )

            regions.append(
                {
                    "name": "本文",
                    "type": "body",
                    "x": int(min_x),
                    "y": int(min_y),
                    "width": int(max_x - min_x),
                    "height": int(max_y - min_y),
                    "orientation": "vertical",
                    "ocr_count": len(main_group)
                }
            )

    # -----------------------------------------------------
    # 横書き本文
    # -----------------------------------------------------

    if horizontal_results:

        horizontal_results.sort(
            key=lambda r: r["y"]
        )

        groups = []

        current_group = []

        line_gap = 20

        for r in horizontal_results:

            if not current_group:
                current_group.append(r)
                continue

            previous = current_group[-1]

            gap = abs(
                r["y"] - previous["y"]
            )

            if gap <= line_gap:
                current_group.append(r)

            else:
                groups.append(
                    current_group
                )

                current_group = [r]

        if current_group:
            groups.append(
                current_group
            )

        # -------------------------------------------------
        # 横書きOCRがまとまっている場合
        # -------------------------------------------------

        if groups:

            groups.sort(
                key=lambda g: len(g),
                reverse=True
            )

            main_group = groups[0]

            min_x = min(
                r["x"]
                for r in main_group
            )

            min_y = min(
                r["y"]
                for r in main_group
            )

            max_x = max(
                r["x"] + r["width"]
                for r in main_group
            )

            max_y = max(
                r["y"] + r["height"]
                for r in main_group
            )

            regions.append(
                {
                    "name": "本文横",
                    "type": "body",
                    "x": int(min_x),
                    "y": int(min_y),
                    "width": int(max_x - min_x),
                    "height": int(max_y - min_y),
                    "orientation": "horizontal",
                    "ocr_count": len(main_group)
                }
            )

    return regions
