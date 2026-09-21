import sys
import json
import subprocess
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import re

# Windows環境における日本語/Unicodeファイルパス対応 (OpenCV cv2.imread / cv2.imwrite)
_orig_cv2_imread = cv2.imread
_orig_cv2_imwrite = cv2.imwrite

def _unicode_safe_imread(filename, flags=cv2.IMREAD_COLOR):
    try:
        p_str = str(filename)
        if all(ord(c) < 128 for c in p_str):
            res = _orig_cv2_imread(p_str, flags)
            if res is not None:
                return res
        with open(p_str, 'rb') as f:
            buf = f.read()
        return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), flags)
    except Exception:
        return None

def _unicode_safe_imwrite(filename, img, params=None):
    try:
        p_str = str(filename)
        if all(ord(c) < 128 for c in p_str):
            res = _orig_cv2_imwrite(p_str, img, params)
            if res:
                return True
        ext = Path(p_str).suffix or '.png'
        success, buf = cv2.imencode(ext, img, params)
        if success:
            with open(p_str, 'wb') as f:
                f.write(buf)
            return True
        return False
    except Exception:
        return False

cv2.imread = _unicode_safe_imread
cv2.imwrite = _unicode_safe_imwrite

# =========================================================
# NDLOCR-Lite JSONを探す
# =========================================================

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

# =========================================================
# 画像前処理（傾き補正、影・ムラ除去、コントラスト強調）
# =========================================================

def detect_and_correct_skew(image: np.ndarray):
    """
    画像の傾き（スキュー角）を検出し、水平・垂直方向に補正する。
    補正した画像、検出角度(度)、逆変換行列(M_inv)を返す。
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # エッジ検出
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # ハフ変換による線分検出
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=max(50, width // 15),
        maxLineGap=10
    )

    angles = []
    if lines is not None:
        for l in lines:
            line = l.flatten()
            if len(line) >= 4:
                x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    continue
                angle = float(np.degrees(np.arctan2(dy, dx)))
                # 水平に近い線分（-15度〜+15度）のみを対象
                if -15.0 <= angle <= 15.0:
                    angles.append(angle)

    # 輪郭のminAreaRectからも角度候補を抽出
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), rect_angle = rect
        if w < 20 or h < 8:
            continue
        if w < h:
            rect_angle = rect_angle + 90
        if rect_angle > 45:
            rect_angle -= 90
        elif rect_angle < -45:
            rect_angle += 90
        if -15.0 <= rect_angle <= 15.0:
            angles.append(rect_angle)

    if not angles:
        return image, 0.0, None

    median_angle = float(np.median(angles))

    # 微小な傾き（0.25度未満）や過度な角度（15度超）は補正スキップ
    if abs(median_angle) < 0.25 or abs(median_angle) > 15.0:
        return image, 0.0, None

    print(f"[前処理] 傾き角検出: {median_angle:.2f}度 - 自動回転補正を実行します", flush=True)

    center = (width / 2.0, height / 2.0)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    M_inv = cv2.getRotationMatrix2D(center, -median_angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        M,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return rotated, median_angle, M_inv


def dewarp_wavy_textlines(image: np.ndarray, orientation: str = "horizontal"):
    """
    紙面の波打ち歪み・見開き本のノド元（綴じ部）湾曲を自動検出し、
    多項式スプラインによる2Dリマップで水平・直線に平坦化（Dewarping）する。
    補正後画像と変位ベクトル（disp_y または disp_x）を返す。
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # 適応的二値化
    block_size = max(15, (min(h, w) // 100) * 2 + 1)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 10)

    if orientation != "vertical":
        # 横書き: 水平方向に文字を連結して行ストリップを形成
        k_w = max(25, w // 70)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_curves = []
        min_line_len = w * 0.10

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < min_line_len or ch > h * 0.15 or ch < 5:
                continue

            pts = cnt.reshape(-1, 2)
            x_unique = np.unique(pts[:, 0])
            if len(x_unique) < 10:
                continue

            sampled_x = []
            sampled_y = []
            step = max(1, len(x_unique) // 30)
            for x_val in x_unique[::step]:
                y_vals = pts[pts[:, 0] == x_val, 1]
                if len(y_vals) > 0:
                    sampled_x.append(x_val)
                    sampled_y.append(float(np.mean(y_vals)))

            if len(sampled_x) >= 6:
                try:
                    poly = np.polyfit(sampled_x, sampled_y, 3)
                    linear = np.polyfit(sampled_x, sampled_y, 1)
                    y_poly = np.polyval(poly, sampled_x)
                    y_lin = np.polyval(linear, sampled_x)
                    dev = float(np.max(np.abs(y_poly - y_lin)))
                    if dev > 1.5:  # 有意な湾曲
                        line_curves.append((sampled_x, poly, linear, dev))
                except Exception:
                    pass

        if not line_curves:
            return image, None, None

        print(f"[前処理] 波打ち行検出: {len(line_curves)}行の湾曲曲線を解析中...", flush=True)

        all_x = np.arange(w)
        total_disp = np.zeros(w, dtype=np.float32)
        counts = np.zeros(w, dtype=np.float32)

        for sx, poly, linear, dev in line_curves:
            curv = np.polyval(poly, all_x) - np.polyval(linear, all_x)
            min_x, max_x = min(sx), max(sx)
            total_disp[min_x:max_x] += curv[min_x:max_x]
            counts[min_x:max_x] += 1.0

        counts[counts == 0] = 1.0
        avg_disp = total_disp / counts
        # ガウシアンで滑らかに補間
        avg_disp = cv2.GaussianBlur(avg_disp.reshape(1, -1), (0, 0), 25.0).flatten()

        max_wave = float(np.max(np.abs(avg_disp)))
        if max_wave < 2.0:
            return image, None, None

        print(f"[前処理] 波打ち歪み検出: 最大振幅 {max_wave:.2f}px - 非線形Dewarping補正を実行します", flush=True)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_y = (grid_y + avg_disp.reshape(1, -1)).astype(np.float32)
        map_x = grid_x.astype(np.float32)

        dewarped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return dewarped, avg_disp.tolist(), None

    else:
        # 縦書き: 垂直方向に文字を連結して列ストリップを形成
        k_h = max(25, h // 70)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col_curves = []
        min_col_len = h * 0.10

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < min_col_len or cw > w * 0.15 or cw < 5:
                continue

            pts = cnt.reshape(-1, 2)
            y_unique = np.unique(pts[:, 1])
            if len(y_unique) < 10:
                continue

            sampled_x = []
            sampled_y = []
            step = max(1, len(y_unique) // 30)
            for y_val in y_unique[::step]:
                x_vals = pts[pts[:, 1] == y_val, 0]
                if len(x_vals) > 0:
                    sampled_y.append(y_val)
                    sampled_x.append(float(np.mean(x_vals)))

            if len(sampled_y) >= 6:
                try:
                    poly = np.polyfit(sampled_y, sampled_x, 3)
                    linear = np.polyfit(sampled_y, sampled_x, 1)
                    x_poly = np.polyval(poly, sampled_y)
                    x_lin = np.polyval(linear, sampled_y)
                    dev = float(np.max(np.abs(x_poly - x_lin)))
                    if dev > 1.5:
                        col_curves.append((sampled_y, poly, linear, dev))
                except Exception:
                    pass

        if not col_curves:
            return image, None, None

        print(f"[前処理] 縦書き波打ち列検出: {len(col_curves)}列の湾曲曲線を解析中...", flush=True)

        all_y = np.arange(h)
        total_disp = np.zeros(h, dtype=np.float32)
        counts = np.zeros(h, dtype=np.float32)

        for sy, poly, linear, dev in col_curves:
            curv = np.polyval(poly, all_y) - np.polyval(linear, all_y)
            min_y, max_y = min(sy), max(sy)
            total_disp[min_y:max_y] += curv[min_y:max_y]
            counts[min_y:max_y] += 1.0

        counts[counts == 0] = 1.0
        avg_disp = total_disp / counts
        avg_disp = cv2.GaussianBlur(avg_disp.reshape(1, -1), (0, 0), 25.0).flatten()

        max_wave = float(np.max(np.abs(avg_disp)))
        if max_wave < 2.0:
            return image, None, None

        print(f"[前処理] 縦書き波打ち歪み検出: 最大振幅 {max_wave:.2f}px - 非線形Dewarping補正を実行します", flush=True)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = (grid_x + avg_disp.reshape(-1, 1)).astype(np.float32)
        map_y = grid_y.astype(np.float32)

        dewarped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return dewarped, None, avg_disp.tolist()


def flatten_illumination_and_shadows(image: np.ndarray) -> np.ndarray:
    """
    本のノド元（綴じ代）の影や不均一な照明ムラを物理的背景除算で平坦化し、
    文字の太さやエッジを一切損なわずに均一な白背景を生成する。
    """
    if len(image.shape) == 2:
        gray = image.copy()
        is_color = False
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True

    h, w = gray.shape[:2]
    k_size = max(31, (min(h, w) // 40) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    # モルフォロジー膨張で文字を埋めて背景輝度マップを推定
    dilated = cv2.dilate(gray, kernel)
    bg = cv2.medianBlur(dilated, 21)

    # 物理的背景除算（Background Division）: (image / bg) * 255.0
    bg_f = bg.astype(np.float32) + 1.0

    if not is_color:
        norm_gray = np.clip((gray.astype(np.float32) / bg_f) * 255.0, 0, 255).astype(np.uint8)
        return norm_gray

    channels = cv2.split(image)
    norm_channels = []
    for ch in channels:
        ch_norm = np.clip((ch.astype(np.float32) / bg_f) * 255.0, 0, 255).astype(np.uint8)
        norm_channels.append(ch_norm)

    return cv2.merge(norm_channels)


def enhance_contrast_and_edges(image: np.ndarray) -> np.ndarray:
    """
    ラプラシアン・高周波アンシャープマスクにより、
    文字のにじみ（ぼやけ）を解消し、微細な漢字の部首や英字輪郭を鮮明化する。
    """
    if len(image.shape) == 2:
        laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
        sharpened = np.clip(image.astype(np.float32) - 0.25 * laplacian, 0, 255).astype(np.uint8)
        return sharpened
    else:
        # BGR画像の場合
        gaussian = cv2.GaussianBlur(image, (0, 0), 1.5)
        sharpened = cv2.addWeighted(image, 1.3, gaussian, -0.3, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)


def transform_point_back(x: float, y: float, transform_info: dict) -> tuple:
    """
    前処理（波打ち歪み補正・傾き回転補正）後の座標 (x, y) を
    元の入力画像（原本PDF）の座標系へ正確に逆変換する。
    """
    if not transform_info:
        return int(round(x)), int(round(y))

    cur_x, cur_y = float(x), float(y)

    # 1. 波打ち補正（Dewarping）の逆変換
    if "disp_y" in transform_info and transform_info["disp_y"]:
        disp_y = transform_info["disp_y"]
        ix = int(round(np.clip(cur_x, 0, len(disp_y) - 1)))
        cur_y = cur_y + disp_y[ix]
    elif "disp_x" in transform_info and transform_info["disp_x"]:
        disp_x = transform_info["disp_x"]
        iy = int(round(np.clip(cur_y, 0, len(disp_x) - 1)))
        cur_x = cur_x + disp_x[iy]

    # 2. 傾き回転補正（M_inv）の逆変換
    if "M_inv" in transform_info and transform_info["M_inv"]:
        M_inv = np.array(transform_info["M_inv"], dtype=np.float32)
        pt = np.array([[[cur_x, cur_y]]], dtype=np.float32)
        trans = cv2.transform(pt, M_inv).reshape(2)
        cur_x, cur_y = float(trans[0]), float(trans[1])

    orig_w = transform_info.get("orig_width", 100000)
    orig_h = transform_info.get("orig_height", 100000)

    res_x = int(round(np.clip(cur_x, 0, orig_w)))
    res_y = int(round(np.clip(cur_y, 0, orig_h)))
    return res_x, res_y


def transform_rect_back(x: int, y: int, width: int, height: int, transform_info: dict) -> dict:
    """
    前処理後の矩形 (x, y, width, height) の4隅を逆変換し、
    元画像座標系における外接矩形を返す。
    """
    if not transform_info or (not transform_info.get("disp_y") and not transform_info.get("disp_x") and not transform_info.get("M_inv")):
        return {"x": x, "y": y, "width": width, "height": height}

    corners = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height)
    ]
    trans_corners = [transform_point_back(cx, cy, transform_info) for cx, cy in corners]

    min_x = min(p[0] for p in trans_corners)
    max_x = max(p[0] for p in trans_corners)
    min_y = min(p[1] for p in trans_corners)
    max_y = max(p[1] for p in trans_corners)

    orig_w = transform_info.get("orig_width", 100000)
    orig_h = transform_info.get("orig_height", 100000)

    min_x = int(round(np.clip(min_x, 0, orig_w)))
    max_x = int(round(np.clip(max_x, 0, orig_w)))
    min_y = int(round(np.clip(min_y, 0, orig_h)))
    max_y = int(round(np.clip(max_y, 0, orig_h)))

    return {
        "x": min_x,
        "y": min_y,
        "width": max(1, max_x - min_x),
        "height": max(1, max_y - min_y)
    }


def is_spread_image(image: np.ndarray) -> tuple:
    """
    見開き（2ページ横並び）画像かどうかを判定し、中央の分割線X座標を返す。
    """
    h, w = image.shape[:2]
    mid = w // 2
    if w < h * 1.05:
        return False, mid
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    search_start = int(w * 0.42)
    search_end = int(w * 0.58)
    
    # ページ間のノド元（余白）はインク量が最小（最も明るい）の縦帯
    ink_density = np.sum(np.maximum(0.0, 240.0 - gray[:, search_start:search_end].astype(np.float32)), axis=0)
    ink_smooth = cv2.GaussianBlur(ink_density.reshape(1, -1), (25, 1), 5.0).flatten()
    
    best_x = search_start + int(np.argmin(ink_smooth))
    if abs(best_x - mid) > int(w * 0.06):
        best_x = mid
        
    return True, best_x


def preprocess_image_array(image: np.ndarray, output_path: Path, orientation_mode: str = "auto", doc_type: str = "japanese") -> tuple:
    """
    メモリ上の画像配列に対するOCR前処理パイプライン：
    1. 傾き検出・自動回転補正
    2. 本の綴じ代の影・不均一照明除去（物理的背景除算）
    3. 波打ち歪み・ノド元湾曲の非線形Dewarping補正
    4. 高周波アンシャープマスクによる文字エッジ鮮鋭化（にじみ解消）
    """
    h, w = image.shape[:2]
    deskewed, angle, M_inv = detect_and_correct_skew(image)
    flattened = flatten_illumination_and_shadows(deskewed)
    dewarped, disp_y, disp_x = dewarp_wavy_textlines(flattened, orientation="vertical" if orientation_mode == "vertical" else "horizontal")
    enhanced = enhance_contrast_and_edges(dewarped)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), enhanced)
    
    transform_info = {
        "angle": angle,
        "M_inv": M_inv.tolist() if M_inv is not None else None,
        "disp_y": disp_y,
        "disp_x": disp_x,
        "orig_width": w,
        "orig_height": h,
    }
    return output_path, transform_info


def preprocess_image_for_ocr(image_path: Path, output_dir: Path, orientation_mode: str = "auto", doc_type: str = "japanese"):
    """
    OCR認識用の入力画像前処理パイプライン
    """
    print()
    print("========== 画像前処理開始 ==========", flush=True)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[警告] 前処理用画像の読み込み失敗: {image_path}", flush=True)
        return image_path, {}

    h, w = image.shape[:2]
    print(f"入力画像サイズ: {w}x{h}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_path = output_dir / "preprocessed_input.png"
    result_path, transform_info = preprocess_image_array(image, preprocessed_path, orientation_mode, doc_type)

    print(f"前処理済み画像保存: {result_path}", flush=True)
    print("========== 画像前処理終了 ==========", flush=True)

    return result_path, transform_info

# =========================================================
# 罫線検出テスト
# =========================================================

def detect_lines(
    image_path: Path,
    output_dir: Path
):
    """
    元画像から横罫線・縦罫線を検出する。

    現段階では、
    「表」「コラム」「図」などの分類は行わない。

    あくまで、
    画像上の直線を正しく検出できるかを確認する。
    """

    print()
    print("========== 罫線検出開始 ==========")

    # -----------------------------------------------------
    # 画像読み込み
    # -----------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"画像を読み込めません: {image_path}"
        )

    height, width = image.shape[:2]

    print(
        f"罫線検出画像サイズ: "
        f"{width}×{height}",
        flush=True
    )

    # -----------------------------------------------------
    # グレースケール
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_path = (
        output_dir
        / "line_gray.png"
    )

    cv2.imwrite(
        str(gray_path),
        gray
    )

    # -----------------------------------------------------
    # 二値化
    #
    # 白背景 → 0
    # 黒い文字・罫線 → 255
    # -----------------------------------------------------

    binary = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_MEAN_C,

        cv2.THRESH_BINARY_INV,

        15,

        10
    )

    binary_path = (
        output_dir
        / "line_binary.png"
    )

    cv2.imwrite(
        str(binary_path),
        binary
    )

    # -----------------------------------------------------
    # 横罫線検出
    # -----------------------------------------------------

    horizontal_kernel_length = max(
        10,
        width // 20
    )

    horizontal_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (
            horizontal_kernel_length,
            1
        )
    )

    horizontal_lines = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        horizontal_kernel
    )

    horizontal_path = (
        output_dir
        / "horizontal_lines.png"
    )

    cv2.imwrite(
        str(horizontal_path),
        horizontal_lines
    )

    # -----------------------------------------------------
    # 縦罫線検出
    # -----------------------------------------------------

    vertical_kernel_length = max(
        10,
        height // 20
    )

    vertical_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (
            1,
            vertical_kernel_length
        )
    )

    vertical_lines = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        vertical_kernel
    )

    vertical_path = (
        output_dir
        / "vertical_lines.png"
    )

    cv2.imwrite(
        str(vertical_path),
        vertical_lines
    )

    # -----------------------------------------------------
    # 横＋縦
    # -----------------------------------------------------

    detected_lines = cv2.bitwise_or(

        horizontal_lines,

        vertical_lines
    )

    detected_path = (
        output_dir
        / "detected_lines.png"
    )

    cv2.imwrite(
        str(detected_path),
        detected_lines
    )

    # -----------------------------------------------------
    # 元画像に罫線を重ねる
    # -----------------------------------------------------

    overlay = image.copy()

    # 検出された横線
    overlay[horizontal_lines > 0] = (
        0,
        0,
        255
    )

    # 検出された縦線
    overlay[vertical_lines > 0] = (
        255,
        0,
        0
    )

    overlay_path = (
        output_dir
        / "detected_lines_overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    # -----------------------------------------------------
    # 検出数を計算
    # -----------------------------------------------------

    horizontal_pixels = int(
        cv2.countNonZero(
            horizontal_lines
        )
    )

    vertical_pixels = int(
        cv2.countNonZero(
            vertical_lines
        )
    )

    total_pixels = int(
        cv2.countNonZero(
            detected_lines
        )
    )

    print(
        f"横罫線画素数: "
        f"{horizontal_pixels}",
        flush=True
    )

    print(
        f"縦罫線画素数: "
        f"{vertical_pixels}",
        flush=True
    )

    print(
        f"罫線総画素数: "
        f"{total_pixels}",
        flush=True
    )

    print(
        f"横罫線画像: "
        f"{horizontal_path}",
        flush=True
    )

    print(
        f"縦罫線画像: "
        f"{vertical_path}",
        flush=True
    )

    print(
        f"重ね合わせ画像: "
        f"{overlay_path}",
        flush=True
    )

    print(
        "========== 罫線検出終了 ==========",
        flush=True
    )

    return {
        "horizontal_pixels": horizontal_pixels,
        "vertical_pixels": vertical_pixels,
        "total_pixels": total_pixels,
        "horizontal_image": str(horizontal_path),
        "vertical_image": str(vertical_path),
        "overlay_image": str(overlay_path)
    }

# =========================================================
# 検出した罫線の座標解析
# =========================================================

def analyze_detected_lines(
    image_path: Path,
    output_dir: Path
):
    """
    横罫線・縦罫線を線分として解析する。

    現段階では矩形判定を行わない。
    検出された線の位置・長さを取得し、
    次段階の矩形生成に利用する。
    """

    print()
    print("========== 罫線座標解析開始 ==========")

    # -----------------------------------------------------
    # 画像読み込み
    # -----------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"画像を読み込めません: {image_path}"
        )

    image_height, image_width = image.shape[:2]

    # -----------------------------------------------------
    # グレースケール
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # 二値化
    # -----------------------------------------------------

    binary = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_MEAN_C,

        cv2.THRESH_BINARY_INV,

        15,

        10
    )

    # -----------------------------------------------------
    # 横罫線検出
    # -----------------------------------------------------

    horizontal_kernel_length = max(
        10,
        image_width // 20
    )

    horizontal_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (
            horizontal_kernel_length,
            1
        )
    )

    horizontal_lines = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        horizontal_kernel
    )

    # -----------------------------------------------------
    # 縦罫線検出
    # -----------------------------------------------------

    vertical_kernel_length = max(
        10,
        image_height // 20
    )

    vertical_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (
            1,
            vertical_kernel_length
        )
    )

    vertical_lines = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        vertical_kernel
    )

    # =====================================================
    # 横線の線分抽出
    # =====================================================

    horizontal_contours, _ = cv2.findContours(

        horizontal_lines,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    horizontal_segments = []

    for contour in horizontal_contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # 短すぎる線は除外
        if w < 10:
            continue

        horizontal_segments.append(

            {
                "x1": int(x),
                "y": int(y),
                "x2": int(x + w - 1),
                "y2": int(y),
                "length": int(w),
                "thickness": int(h)
            }
        )

    # Y座標 → X座標順
    horizontal_segments.sort(

        key=lambda line: (
            line["y"],
            line["x1"]
        )
    )

    # =====================================================
    # 縦線の線分抽出
    # =====================================================

    vertical_contours, _ = cv2.findContours(

        vertical_lines,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    vertical_segments = []

    for contour in vertical_contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # 短すぎる線は除外
        if h < 10:
            continue

        vertical_segments.append(

            {
                "x": int(x),
                "y1": int(y),
                "x2": int(x),
                "y2": int(y + h - 1),
                "length": int(h),
                "thickness": int(w)
            }
        )

    # X座標 → Y座標順
    vertical_segments.sort(

        key=lambda line: (
            line["x"],
            line["y1"]
        )
    )

    # =====================================================
    # 結果表示
    # =====================================================

    print()
    print(
        f"横線候補数: "
        f"{len(horizontal_segments)}"
    )

    for i, line in enumerate(
        horizontal_segments
    ):

        print(

            f"  H{i + 1}: "

            f"X={line['x1']}～{line['x2']}, "

            f"Y={line['y']}, "

            f"長さ={line['length']}",

            flush=True
        )

    print()

    print(
        f"縦線候補数: "
        f"{len(vertical_segments)}"
    )

    for i, line in enumerate(
        vertical_segments
    ):

        print(

            f"  V{i + 1}: "

            f"X={line['x']}, "

            f"Y={line['y1']}～{line['y2']}, "

            f"長さ={line['length']}",

            flush=True
        )

    # =====================================================
    # JSON保存
    # =====================================================

    result = {

        "image": str(
            image_path
        ),

        "image_width": int(
            image_width
        ),

        "image_height": int(
            image_height
        ),

        "horizontal_lines":
            horizontal_segments,

        "vertical_lines":
            vertical_segments
    }

    output_json = (
        output_dir
        / "detected_line_segments.json"
    )

    with output_json.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            result,

            f,

            ensure_ascii=False,

            indent=2
        )

    # =====================================================
    # 確認画像
    # =====================================================

    overlay = image.copy()

    # -----------------------------------------------------
    # 横線：赤
    # -----------------------------------------------------

    for line in horizontal_segments:

        cv2.line(

            overlay,

            (
                line["x1"],
                line["y"]
            ),

            (
                line["x2"],
                line["y2"]
            ),

            (0, 0, 255),

            2
        )

    # -----------------------------------------------------
    # 縦線：青
    # -----------------------------------------------------

    for line in vertical_segments:

        cv2.line(

            overlay,

            (
                line["x"],
                line["y1"]
            ),

            (
                line["x2"],
                line["y2"]
            ),

            (255, 0, 0),

            2
        )

    overlay_path = (
        output_dir
        / "line_segments_overlay.png"
    )

    cv2.imwrite(

        str(overlay_path),

        overlay
    )

    print()
    print(
        f"線分JSON: "
        f"{output_json}"
    )

    print(
        f"線分確認画像: "
        f"{overlay_path}"
    )

    print(
        "========== 罫線座標解析終了 ==========",
        flush=True
    )

    return result

# =========================================================
# 罫線から矩形領域を検出
# =========================================================

def detect_bordered_regions(
    image_path: Path,
    output_dir: Path
):
    """
    横罫線・縦罫線から閉じた矩形領域を検出する。

    現段階では「表」とは判定しない。
    あくまで「罫線によって囲まれた領域」を検出する。
    """

    print()
    print("========== 罫線領域検出開始 ==========")

    # -----------------------------------------------------
    # 画像読み込み
    # -----------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"画像を読み込めません: {image_path}"
        )

    image_height, image_width = image.shape[:2]

    # -----------------------------------------------------
    # グレースケール
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # 二値化
    # -----------------------------------------------------

    binary = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_MEAN_C,

        cv2.THRESH_BINARY_INV,

        15,

        10
    )

    # -----------------------------------------------------
    # 横罫線
    # -----------------------------------------------------

    horizontal_kernel_length = max(
        10,
        image_width // 20
    )

    horizontal_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (
            horizontal_kernel_length,
            1
        )
    )

    horizontal_lines = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        horizontal_kernel
    )

    # -----------------------------------------------------
    # 縦罫線
    # -----------------------------------------------------

    vertical_kernel_length = max(
        10,
        image_height // 20
    )

    vertical_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (
            1,
            vertical_kernel_length
        )
    )

    vertical_lines = cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        vertical_kernel
    )

    # -----------------------------------------------------
    # 横＋縦
    # -----------------------------------------------------

    line_image = cv2.bitwise_or(

        horizontal_lines,

        vertical_lines
    )

    # -----------------------------------------------------
    # 線を少し太くする
    #
    # 交差部分を確実につなげるため
    # -----------------------------------------------------

    connect_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (3, 3)
    )

    connected_lines = cv2.dilate(

        line_image,

        connect_kernel,

        iterations=1
    )

    # -----------------------------------------------------
    # 輪郭検出
    # -----------------------------------------------------

    contours, hierarchy = cv2.findContours(

        connected_lines,

        cv2.RETR_LIST,

        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    # -----------------------------------------------------
    # 矩形候補を取得
    # -----------------------------------------------------

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # ---------------------------------------------
        # 小さすぎる領域を除外
        # ---------------------------------------------

        if w < 30 or h < 20:
            continue

        # ---------------------------------------------
        # 画像全体に近いものを除外
        #
        # ページ全体を囲むような輪郭を
        # 罫線領域として扱わない
        # ---------------------------------------------

        area_ratio = (
            (w * h)
            /
            (image_width * image_height)
        )

        if area_ratio > 0.90:
            continue

        # ---------------------------------------------
        # 横線・縦線が実際に存在するか確認
        # ---------------------------------------------

        roi_horizontal = horizontal_lines[
            y:min(y + h, image_height),
            x:min(x + w, image_width)
        ]

        roi_vertical = vertical_lines[
            y:min(y + h, image_height),
            x:min(x + w, image_width)
        ]

        horizontal_pixels = cv2.countNonZero(
            roi_horizontal
        )

        vertical_pixels = cv2.countNonZero(
            roi_vertical
        )

        # ---------------------------------------------
        # 横線と縦線の両方が存在する領域だけ採用
        # ---------------------------------------------

        if horizontal_pixels == 0:
            continue

        if vertical_pixels == 0:
            continue

        # ---------------------------------------------
        # 重複チェック用
        # ---------------------------------------------

        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(w * h),
                "horizontal_pixels": int(
                    horizontal_pixels
                ),
                "vertical_pixels": int(
                    vertical_pixels
                )
            }
        )

    # -----------------------------------------------------
    # 小さい矩形を内包する大きな矩形がある場合、
    # まずはすべて保存する。
    #
    # 後段で「表全体」に統合する。
    # -----------------------------------------------------

    regions.sort(
        key=lambda r: (
            r["y"],
            r["x"],
            -r["area"]
        )
    )

    # -----------------------------------------------------
    # JSON保存
    # -----------------------------------------------------

    bordered_result = {

        "image": str(
            image_path
        ),

        "image_width": int(
            image_width
        ),

        "image_height": int(
            image_height
        ),

        "regions": regions
    }

    bordered_json = (
        output_dir
        / "bordered_regions.json"
    )

    with bordered_json.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            bordered_result,

            f,

            ensure_ascii=False,

            indent=2
        )

    # -----------------------------------------------------
    # 確認用画像
    # -----------------------------------------------------

    overlay = image.copy()

    for i, region in enumerate(regions):

        x = region["x"]
        y = region["y"]
        w = region["width"]
        h = region["height"]

        cv2.rectangle(

            overlay,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2
        )

        cv2.putText(

            overlay,

            f"R{i + 1}",

            (x, max(15, y - 5)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            1,

            cv2.LINE_AA
        )

    overlay_path = (
        output_dir
        / "bordered_regions_overlay.png"
    )

    cv2.imwrite(

        str(overlay_path),

        overlay
    )

    # -----------------------------------------------------
    # 結果表示
    # -----------------------------------------------------

    print(
        f"罫線矩形候補数: "
        f"{len(regions)}",
        flush=True
    )

    print(
        f"罫線領域JSON: "
        f"{bordered_json}",
        flush=True
    )

    print(
        f"罫線領域確認画像: "
        f"{overlay_path}",
        flush=True
    )

    for i, region in enumerate(regions):

        print(

            f"  R{i + 1}: "

            f"x={region['x']}, "

            f"y={region['y']}, "

            f"width={region['width']}, "

            f"height={region['height']}",

            flush=True
        )

    print(
        "========== 罫線領域検出終了 ==========",
        flush=True
    )

    return regions

# =========================================================
# NDLOCR-Lite JSONを解析
# =========================================================

def clean_runaway_repetition(text: str) -> str:
    if not text:
        return text
    # 5回以上連続する同一文字（00000, 11111, ......, ------等）以降の暴走出力を除去
    text = re.sub(r'([^\s])\1{4,}.*$', '', text).strip()
    # 繰り返し単語の暴走（the the the the 等）を除去
    text = re.sub(r'(\b\w+\b\s+)\1{3,}.*$', '', text).strip()
    # 年号直後の括弧誤認識 (例: 2005al. -> 2005a]., 1997l. -> 1997].)
    text = re.sub(r'(\b\d{4}[a-z]?)l\.', r'\1].', text)
    text = re.sub(r'(\b\d{4}[a-z]?)1\.', r'\1].', text)
    return text

def parse_ndlocr_json(json_path: Path, transform_info: dict = None):

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

        # -------------------------------------------------
        # 空白・微小ノイズ・低信頼度ノイズの除外（ページの端への誤拡張防止）
        # -------------------------------------------------
        text_clean = text.strip()
        if not text_clean:
            continue
        if len(text_clean) == 1 and text_clean in "._,-~|'`\"^*:;・+=/\\()[]{}<>" and confidence < 0.35:
            continue
        if width < 8 or height < 8:
            continue
        if 0.0 < confidence < 0.15:
            continue
        if confidence == 0.0 and len(text_clean) <= 1:
            continue

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

    if transform_info:
        for r in results:
            rect = transform_rect_back(r["x"], r["y"], r["width"], r["height"], transform_info)
            r["x"] = rect["x"]
            r["y"] = rect["y"]
            r["width"] = rect["width"]
            r["height"] = rect["height"]

    # 重複・包含行の統合（text_block由来のconf=0.0行と個別line_main行が共存する場合など）
    to_remove = set()
    n = len(results)
    for i in range(n):
        if i in to_remove:
            continue
        a = results[i]
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            b = results[j]
            v_inter = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
            v_max = max(a["height"], b["height"])
            if v_max <= 0:
                continue
            h_inter = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
            h_max = max(a["width"], b["width"])
            if h_max <= 0:
                continue

            if a.get("isVertical", False) or b.get("isVertical", False):
                l_ratio = v_inter / v_max
                w_ratio = h_inter / h_max
            else:
                l_ratio = h_inter / h_max
                w_ratio = v_inter / v_max

            if w_ratio > 0.4 and l_ratio > 0.5:
                # 比較時は空白を除いた文字数で比較（英数字間のスペースによる文字数の水増しを防止）
                clean_len_a = len(re.sub(r'\s+', '', a["text"]))
                clean_len_b = len(re.sub(r'\s+', '', b["text"]))
                max_conf = max(a.get("confidence", 0.0), b.get("confidence", 0.0))

                if clean_len_b > clean_len_a + 3:
                    b["confidence"] = max_conf
                    to_remove.add(i)
                    break
                elif clean_len_a > clean_len_b + 3:
                    a["confidence"] = max_conf
                    to_remove.add(j)
                else:
                    if b.get("confidence", 0.0) > a.get("confidence", 0.0) and b.get("confidence", 0.0) > 0:
                        b["confidence"] = max_conf
                        to_remove.add(i)
                        break
                    else:
                        a["confidence"] = max_conf
                        to_remove.add(j)

    results = [results[i] for i in range(n) if i not in to_remove]
    return results

# =========================================================
# 線分から罫線領域を生成
# =========================================================

def create_bordered_regions_from_lines(
    line_data,
    image_width,
    image_height
):
    """
    検出された横線・縦線から
    罫線で囲まれた大きな領域を生成する。

    現段階では「表」とは判定しない。
    type は bordered とする。
    """

    horizontal_lines = (
        line_data["horizontal_lines"]
    )

    vertical_lines = (
        line_data["vertical_lines"]
    )

    # 十分に長い横線
    horizontal_candidates = [
        line
        for line in horizontal_lines
        if line["length"] >= max(30, image_width * 0.25)
    ]

    # 十分に長い縦線
    vertical_candidates = [
        line
        for line in vertical_lines
        if line["length"] >= max(20, image_height * 0.15)
    ]

    print()
    print(
        "========== 罫線領域生成 =========="
    )

    print(
        f"長い横線候補: "
        f"{len(horizontal_candidates)}"
    )

    for line in horizontal_candidates:

        print(

            f"  H: "
            f"X={line['x1']}～{line['x2']}, "
            f"Y={line['y']}, "
            f"L={line['length']}"
        )

    print(
        f"長い縦線候補: "
        f"{len(vertical_candidates)}"
    )

    for line in vertical_candidates:

        print(

            f"  V: "
            f"X={line['x']}, "
            f"Y={line['y1']}～{line['y2']}, "
            f"L={line['length']}"
        )

    regions = []

    # -----------------------------------------------------
    # 横線2本の組み合わせ
    # -----------------------------------------------------

    for i in range(
        len(horizontal_candidates)
    ):

        top = horizontal_candidates[i]

        for j in range(
            i + 1,
            len(horizontal_candidates)
        ):

            bottom = horizontal_candidates[j]

            # 上下関係
            if bottom["y"] <= top["y"]:
                continue

            # -------------------------------------------------
            # 上線と下線の間隔
            # -------------------------------------------------

            region_height = (
                bottom["y"]
                - top["y"]
            )

            if region_height < 30:
                continue

            # -------------------------------------------------
            # 横線の共通範囲
            # -------------------------------------------------

            common_x1 = max(
                top["x1"],
                bottom["x1"]
            )

            common_x2 = min(
                top["x2"],
                bottom["x2"]
            )

            if common_x2 <= common_x1:
                continue

            # -------------------------------------------------
            # この上下線をつなぐ縦線を探す
            # -------------------------------------------------

            left_vertical = None
            right_vertical = None

            for vertical in vertical_candidates:

                x = vertical["x"]

                # 縦線が上下の横線の範囲内にあるか
                if x < common_x1:
                    continue

                if x > common_x2:
                    continue

                # 縦線が上下の横線を十分につないでいるか
                if vertical["y1"] > top["y"] + 5:
                    continue

                if vertical["y2"] < bottom["y"] - 5:
                    continue

                # 左端候補
                if left_vertical is None:

                    left_vertical = vertical

                elif x < left_vertical["x"]:

                    left_vertical = vertical

            # -------------------------------------------------
            # 右端の縦線
            # -------------------------------------------------

            for vertical in vertical_candidates:

                x = vertical["x"]

                if x < common_x1:
                    continue

                if x > common_x2:
                    continue

                if vertical["y1"] > top["y"] + 5:
                    continue

                if vertical["y2"] < bottom["y"] - 5:
                    continue

                if right_vertical is None:

                    right_vertical = vertical

                elif x > right_vertical["x"]:

                    right_vertical = vertical

            # -------------------------------------------------
            # 左右の縦線が存在するか
            # -------------------------------------------------

            has_both_verticals = (
                left_vertical is not None
                and right_vertical is not None
                and left_vertical["x"] < right_vertical["x"]
            )

            if has_both_verticals:
                # 縦線が横線の端に近い（外枠）場合のみx1/x2として採用
                # 横線が縦線より外側に大きく伸びている場合（オープン表・三線表で内部縦線のみ存在する場合）：
                # 表全体の幅は横線の端（common_x1, common_x2）を採用し、縦線は内部罫線として扱う
                if left_vertical["x"] - common_x1 > 40:
                    x1 = common_x1
                else:
                    x1 = left_vertical["x"]

                if common_x2 - right_vertical["x"] > 40:
                    x2 = common_x2
                else:
                    x2 = right_vertical["x"]
            else:
                # 左右外枠の縦線がない場合（オープン表／三線表）：
                # 2本以上の横線が大きく重なり、かつ内部に縦線または別の横線が存在する場合を表領域とする
                span_width = common_x2 - common_x1
                if span_width < max(30, image_width * 0.2):
                    continue

                has_internal_v = any(
                    common_x1 - 10 <= v["x"] <= common_x2 + 10 and
                    v["y1"] <= bottom["y"] + 15 and v["y2"] >= top["y"] - 15
                    for v in vertical_candidates
                )
                has_intermediate_h = any(
                    top["y"] + 5 < h["y"] < bottom["y"] - 5 and
                    max(h["x1"], common_x1) < min(h["x2"], common_x2)
                    for h in horizontal_candidates
                )

                if not (has_internal_v or has_intermediate_h):
                    continue

                x1 = common_x1
                x2 = common_x2

            y1 = top["y"]
            y2 = bottom["y"]

            width = x2 - x1
            height = y2 - y1

            if width < 30 or height < 30:
                continue

            regions.append(

                {
                    "name": "罫線領域",
                    "type": "bordered",
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(width),
                    "height": int(height)
                }

            )
    
    # -----------------------------------------------------
    # 重複・内包領域を整理
    # -----------------------------------------------------

    regions = merge_overlapping_bordered_regions(
        regions
    )

    # -----------------------------------------------------
    # 表示
    # -----------------------------------------------------

    print(
        f"生成された罫線領域数: "
        f"{len(regions)}"
    )

    for i, region in enumerate(regions):

        print(
            f"  B{i + 1}: "
            f"x={region['x']}, "
            f"y={region['y']}, "
            f"width={region['width']}, "
            f"height={region['height']}"
        )

    print(
        "===================================="
    )

    return regions


def merge_overlapping_bordered_regions(regions):
    """
    罫線矩形候補から、
    同一表に属する重複・内包矩形を整理する。

    完全に内包されている小矩形は、
    大きな表領域に統合する。

    一方、別位置にある表は別領域として残す。
    """

    if not regions:
        return []

    def contains(outer, inner, margin=5):
        return (
            outer["x"] <= inner["x"] + margin
            and
            outer["y"] <= inner["y"] + margin
            and
            outer["x"] + outer["width"]
            >=
            inner["x"] + inner["width"] - margin
            and
            outer["y"] + outer["height"]
            >=
            inner["y"] + inner["height"] - margin
        )

    # 面積の大きい順
    sorted_regions = sorted(
        regions,
        key=lambda r: r["width"] * r["height"],
        reverse=True
    )

    selected = []

    for region in sorted_regions:

        # 既存の大きな領域に完全に
        # 内包されている場合は除外
        contained = False

        for existing in selected:

            if contains(existing, region):
                contained = True
                break

        if contained:
            continue

        selected.append(region)

    # ページ上の位置順
    selected.sort(
        key=lambda r: (
            r["y"],
            r["x"]
        )
    )

    return selected

    

    # -----------------------------------------------------
    # 結果
    # -----------------------------------------------------

    print(
        f"生成された罫線領域数: "
        f"{len(regions)}"
    )

    for i, region in enumerate(
        regions
    ):

        print(

            f"  B{i + 1}: "

            f"x={region['x']}, "

            f"y={region['y']}, "

            f"width={region['width']}, "

            f"height={region['height']}"
        )

    print(
        "===================================="
    )

    return regions

# =========================================================
# 罫線領域内部の罫線座標を解析
# =========================================================

def analyze_bordered_region_lines(
    line_data,
    bordered_regions
):
    """
    罫線領域の内部に存在する
    縦罫線・横罫線を整理する。

    現段階ではセル生成や表判定は行わない。
    """

    print()
    print(
        "========== 内部罫線解析開始 =========="
    )

    analyzed_regions = []

    horizontal_lines = line_data.get(
        "horizontal_lines",
        []
    )

    vertical_lines = line_data.get(
        "vertical_lines",
        []
    )

    for region_index, region in enumerate(
        bordered_regions
    ):

        rx1 = region["x"]
        ry1 = region["y"]

        rx2 = (
            rx1
            + region["width"]
        )

        ry2 = (
            ry1
            + region["height"]
        )

        print()
        print(
            f"--- 罫線領域 B{region_index + 1} ---"
        )

        print(
            f"領域: "
            f"X={rx1}～{rx2}, "
            f"Y={ry1}～{ry2}"
        )

        # =================================================
        # 領域内部の横罫線
        # =================================================

        region_horizontal = []

        for line in horizontal_lines:

            y = line["y"]

            if y < ry1 - 3:
                continue

            if y > ry2 + 3:
                continue

            # 横線と領域のX範囲が重なっているか
            overlap_x1 = max(
                line["x1"],
                rx1
            )

            overlap_x2 = min(
                line["x2"],
                rx2
            )

            if overlap_x2 <= overlap_x1:
                continue

            region_horizontal.append(
                {
                    "y": int(y),
                    "x1": int(overlap_x1),
                    "x2": int(overlap_x2),
                    "length": int(
                        overlap_x2
                        - overlap_x1
                        + 1
                    )
                }
            )

        # -------------------------------------------------
        # Y座標の近い線を統合
        # -------------------------------------------------

        region_horizontal.sort(
            key=lambda line: line["y"]
        )

        horizontal_positions = []

        for line in region_horizontal:

            if not horizontal_positions:

                horizontal_positions.append(
                    line["y"]
                )

                continue

            previous_y = (
                horizontal_positions[-1]
            )

            if abs(
                line["y"]
                - previous_y
            ) <= 8:

                # 平均位置
                horizontal_positions[-1] = int(
                    round(
                        (
                            previous_y
                            + line["y"]
                        )
                        / 2
                    )
                )

            else:

                horizontal_positions.append(
                    line["y"]
                )

        # =================================================
        # 領域内部の縦罫線
        # =================================================

        region_vertical = []

        for line in vertical_lines:

            x = line["x"]

            if x < rx1 - 3:
                continue

            if x > rx2 + 3:
                continue

            # 縦線と領域のY範囲が重なっているか
            overlap_y1 = max(
                line["y1"],
                ry1
            )

            overlap_y2 = min(
                line["y2"],
                ry2
            )

            if overlap_y2 <= overlap_y1:
                continue

            region_vertical.append(
                {
                    "x": int(x),
                    "y1": int(overlap_y1),
                    "y2": int(overlap_y2),
                    "length": int(
                        overlap_y2
                        - overlap_y1
                        + 1
                    )
                }
            )

        # -------------------------------------------------
        # X座標の近い線を統合
        # -------------------------------------------------

        region_vertical.sort(
            key=lambda line: line["x"]
        )

        vertical_positions = []

        for line in region_vertical:

            if not vertical_positions:

                vertical_positions.append(
                    line["x"]
                )

                continue

            previous_x = (
                vertical_positions[-1]
            )

            if abs(
                line["x"]
                - previous_x
            ) <= 12:

                vertical_positions[-1] = int(
                    round(
                        (
                            previous_x
                            + line["x"]
                        )
                        / 2
                    )
                )

            else:

                vertical_positions.append(
                    line["x"]
                )

        # 外枠の境界を horizontal_positions / vertical_positions に確実に補完
        if not horizontal_positions or abs(horizontal_positions[0] - ry1) > 8:
            horizontal_positions.insert(0, int(ry1))
        if abs(horizontal_positions[-1] - ry2) > 8:
            horizontal_positions.append(int(ry2))

        if not vertical_positions or abs(vertical_positions[0] - rx1) > 8:
            vertical_positions.insert(0, int(rx1))
        if abs(vertical_positions[-1] - rx2) > 8:
            vertical_positions.append(int(rx2))

        # =================================================
        # 結果表示
        # =================================================

        print()
        print(
            "横罫線Y座標:"
        )

        print(
            horizontal_positions
        )

        print()
        print(
            "縦罫線X座標:"
        )

        print(
            vertical_positions
        )

        # =================================================
        # 結果保存
        # =================================================

        analyzed_regions.append(
            {
                "name": region["name"],
                "type": region["type"],
                "x": rx1,
                "y": ry1,
                "width": region["width"],
                "height": region["height"],
                "horizontal_positions":
                    horizontal_positions,
                "vertical_positions":
                    vertical_positions
            }
        )

    print()
    print(
        "========== 内部罫線解析終了 ==========",
        flush=True
    )

    return analyzed_regions

# =========================================================
# 罫線領域からセル候補を生成
# =========================================================

def create_cell_candidates(
    analyzed_regions
):
    """
    罫線領域のX/Y座標から
    セル候補を生成する。

    現段階ではセルを確定しない。
    あくまで矩形候補として扱う。
    """

    print()
    print(
        "========== セル候補生成開始 =========="
    )

    all_regions = []

    for region_index, region in enumerate(
        analyzed_regions
    ):

        x_positions = region.get(
            "vertical_positions",
            []
        )

        y_positions = region.get(
            "horizontal_positions",
            []
        )

        if len(x_positions) < 2:
            continue

        if len(y_positions) < 2:
            continue

        print()
        print(
            f"--- 罫線領域 B{region_index + 1} ---"
        )

        print(
            f"列数候補: "
            f"{len(x_positions) - 1}"
        )

        print(
            f"行数候補: "
            f"{len(y_positions) - 1}"
        )

        cells = []

        # -------------------------------------------------
        # Y方向
        # -------------------------------------------------

        for row in range(
            len(y_positions) - 1
        ):

            y1 = y_positions[row]
            y2 = y_positions[row + 1]

            # -------------------------------------------------
            # X方向
            # -------------------------------------------------

            for column in range(
                len(x_positions) - 1
            ):

                x1 = x_positions[column]
                x2 = x_positions[column + 1]

                width = x2 - x1
                height = y2 - y1

                if width <= 0:
                    continue

                if height <= 0:
                    continue

                cells.append(
                    {
                        "row": row + 1,
                        "column": column + 1,
                        "x": x1,
                        "y": y1,
                        "width": width,
                        "height": height,

                        # OCR割り当て用
                        "text": "",
                        "ocr_count": 0
                    }
                )                

        # -------------------------------------------------
        # 結果表示
        # -------------------------------------------------

        print(
            f"生成セル候補数: "
            f"{len(cells)}"
        )

        for cell in cells:

            print(
                f"  "
                f"R{cell['row']}C{cell['column']}: "
                f"x={cell['x']}, "
                f"y={cell['y']}, "
                f"w={cell['width']}, "
                f"h={cell['height']}"
            )

        all_regions.append(
            {
                "name": region["name"],
                "type": region["type"],
                "x": region["x"],
                "y": region["y"],
                "width": region["width"],
                "height": region["height"],
                "cells": cells
            }
        )

    print()
    print(
        "========== セル候補生成終了 ==========",
        flush=True
    )

    return all_regions

# =========================================================
# セル候補にOCR結果を割り当てる
# =========================================================

def assign_ocr_to_cell_candidates(
    cell_candidate_regions,
    results
):
    """
    各セル候補にOCR結果を割り当てる。

    OCRの中心点がセル内にある場合、
    そのOCRをセルに所属させる。

    現段階では「表セル」とは判定しない。
    """

    print()
    print(
        "========== セルOCR割り当て開始 =========="
    )

    analyzed_regions = []

    for region_index, region in enumerate(
        cell_candidate_regions
    ):

        print()
        print(
            f"--- 罫線領域 B{region_index + 1} ---"
        )

        analyzed_cells = []

        for cell in region.get(
            "cells",
            []
        ):

            cell_x1 = cell["x"]
            cell_y1 = cell["y"]

            cell_x2 = (
                cell_x1
                + cell["width"]
            )

            cell_y2 = (
                cell_y1
                + cell["height"]
            )

            cell_ocr = []

            for ocr in results:

                center_x = (
                    ocr["x"]
                    + ocr["width"] / 2.0
                )

                center_y = (
                    ocr["y"]
                    + ocr["height"] / 2.0
                )

                if (
                    center_x >= cell_x1
                    and center_x <= cell_x2
                    and center_y >= cell_y1
                    and center_y <= cell_y2
                ):

                    cell_ocr.append(
                        ocr
                    )

            analyzed_cell = dict(
                cell
            )

            analyzed_cell["ocr_count"] = len(cell_ocr)

            analyzed_cell["ocr"] = cell_ocr

            # -------------------------------------------------
            # OCR文字列
            # -------------------------------------------------
            texts = [
                ocr["text"]
                for ocr in cell_ocr
                if ocr.get("text")
            ]

            analyzed_cell["text"] = " ".join(texts)

            analyzed_cells.append(
                analyzed_cell
            )

        analyzed_regions.append(
            {
                "name": region["name"],
                "type": region["type"],
                "x": region["x"],
                "y": region["y"],
                "width": region["width"],
                "height": region["height"],
                "cells": analyzed_cells
            }
        )

        # -------------------------------------------------
        # 表示
        # -------------------------------------------------

        for cell in analyzed_cells:

            if cell["ocr_count"] == 0:
                continue

            texts = [
                ocr["text"]
                for ocr in cell["ocr"]
            ]

            text = " ".join(
                texts
            )

            print(
                f"  "
                f"R{cell['row']}C{cell['column']}: "
                f"OCR={cell['ocr_count']}, "
                f"text={text}"
            )

    print()
    print(
        "========== セルOCR割り当て終了 ==========",
        flush=True
    )

    return analyzed_regions

# =========================================================
# 自動レイアウト領域生成
# =========================================================



# =========================================================
# 罫線領域＋セルOCR結果を表領域に変換
# =========================================================

def create_table_regions_from_cells(
    analyzed_cell_regions
):
    """
    罫線領域とセルOCR結果から
    auto_layout.json用の表regionを生成する。
    """

    regions = []

    # 見開き2ページまたは段組を考慮した自然な読み順（左側ページ/カラム -> 右側ページ/カラム、各列は上から下）に整列
    sorted_tables = sorted(
        analyzed_cell_regions,
        key=lambda t: (0 if t.get("x", 0) < 1650 else 1, t.get("y", 0))
    )

    for table_index, table in enumerate(
        sorted_tables,
        start=1
    ):

        cells = []

        for cell in table.get(
            "cells",
            []
        ):

            texts = [
                ocr["text"]
                for ocr in cell.get(
                    "ocr",
                    []
                )
            ]

            text = " ".join(
                texts
            )

            cells.append(
                {
                    "row": cell["row"],
                    "column": cell["column"],

                    "x": cell["x"],
                    "y": cell["y"],
                    "width": cell["width"],
                    "height": cell["height"],

                    "text": text,

                    "ocr_count": cell.get(
                        "ocr_count",
                        0
                    )
                }
            )

        # 枠内の罫線（横・縦）座標を抽出
        h_lines = set()
        v_lines = set()
        for cell in cells:
            cy1 = cell["y"]
            cy2 = cell["y"] + cell["height"]
            cx1 = cell["x"]
            cx2 = cell["x"] + cell["width"]
            if table["y"] + 2 < cy1 < table["y"] + table["height"] - 2:
                h_lines.add(cy1)
            if table["y"] + 2 < cy2 < table["y"] + table["height"] - 2:
                h_lines.add(cy2)
            if table["x"] + 2 < cx1 < table["x"] + table["width"] - 2:
                v_lines.add(cx1)
            if table["x"] + 2 < cx2 < table["x"] + table["width"] - 2:
                v_lines.add(cx2)

        def cluster_coords(coords, tol=4):
            if not coords:
                return []
            s = sorted(coords)
            clusters = [[s[0]]]
            for val in s[1:]:
                if abs(val - sum(clusters[-1])/len(clusters[-1])) <= tol:
                    clusters[-1].append(val)
                else:
                    clusters.append([val])
            return [int(round(sum(c)/len(c))) for c in clusters]

        horizontal_lines = cluster_coords(h_lines)
        vertical_lines = cluster_coords(v_lines)

        regions.append(
            {
                "name": f"表{table_index}",
                "type": "table",

                "x": table["x"],
                "y": table["y"],
                "width": table["width"],
                "height": table["height"],

                "rows": len(
                    set(
                        cell["row"]
                        for cell in cells
                    )
                ),

                "columns": len(
                    set(
                        cell["column"]
                        for cell in cells
                    )
                ),

                "horizontal_lines": horizontal_lines,
                "vertical_lines": vertical_lines,

                "cells": cells
            }
        )

    return regions



# =========================================================
# 表セル内のOCRを本文判定から除外
# =========================================================

def remove_table_ocr(
    results,
    analyzed_cell_regions
):
    """
    表領域に属するOCRを本文候補から除外する。

    1. セルへ割り当て済みのOCRを除外する。
    2. セル割り当てに失敗したOCRでも、OCR中心点が表領域内なら除外する。
    """

    table_ocr_ids = set()
    table_regions = []

    for region in analyzed_cell_regions:
        table_regions.append({
            "x": region["x"],
            "y": region["y"],
            "width": region["width"],
            "height": region["height"]
        })

        for cell in region.get("cells", []):
            for ocr in cell.get("ocr", []):
                table_ocr_ids.add(id(ocr))

    body_results = []

    for ocr in results:
        # セル割り当て済み
        if id(ocr) in table_ocr_ids:
            continue

        # セル割り当てに失敗していても、表領域内なら除外
        center_x = ocr["x"] + ocr["width"] / 2
        center_y = ocr["y"] + ocr["height"] / 2

        inside_table = False
        for table in table_regions:
            tx1 = table["x"]
            ty1 = table["y"]
            tx2 = tx1 + table["width"]
            ty2 = ty1 + table["height"]

            if tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2:
                inside_table = True
                break

        if inside_table:
            continue

        body_results.append(ocr)

    return body_results

# =========================================================
# 表以外のOCRから本文領域を生成
# =========================================================

def create_body_regions(
    body_results,
    orientation_mode="auto",
    doc_type="japanese"
):
    """
    表セルに割り当てられていないOCR結果から
    本文候補領域を生成する。

    縦書き・横書きの両方に対応する。
    """

    if not body_results:
        return []

    # -----------------------------------------------------
    # OCR結果を縦書き・横書きに分離（設定による強制/自動）
    # -----------------------------------------------------

    if doc_type == "western" or orientation_mode == "horizontal":
        vertical_results = []
        horizontal_results = list(body_results)
    elif orientation_mode == "vertical":
        vertical_results = list(body_results)
        horizontal_results = []
    else:
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

        median_w = np.median([r["width"] for r in vertical_results]) if vertical_results else 30
        column_gap = max(45, int(median_w * 2.2))

        groups = []
        current_group = []

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
        # 各グループを本文領域として登録（見開き・段組・複数ブロック対応）
        # -------------------------------------------------

        for g_idx, group in enumerate(groups):
            if not group:
                continue

            valid_items = [r for r in group if r.get("text", "").strip()]
            if not valid_items:
                valid_items = group

            min_x = min(r["x"] for r in valid_items)
            min_y = min(r["y"] for r in valid_items)
            max_x = max(r["x"] + r["width"] for r in valid_items)
            max_y = max(r["y"] + r["height"] for r in valid_items)

            regions.append(
                {
                    "name": "本文" if len(groups) == 1 else f"本文_{g_idx + 1}",
                    "type": "body",
                    "x": int(min_x),
                    "y": int(min_y),
                    "width": int(max_x - min_x),
                    "height": int(max_y - min_y),
                    "orientation": "vertical",
                    "ocr_count": len(valid_items)
                }
            )

    # -----------------------------------------------------
    # 横書き本文
    # -----------------------------------------------------

    if horizontal_results:
        # 1. 見開き・複数段組の水平分割（左右のページ／段の分離）
        all_min_x = min(r["x"] for r in horizontal_results)
        all_max_x = max(r["x"] + r["width"] for r in horizontal_results)
        total_span_w = all_max_x - all_min_x

        mid_split_x = all_min_x + total_span_w / 2.0
        left_side = [r for r in horizontal_results if (r["x"] + r["width"] / 2.0) < mid_split_x]
        right_side = [r for r in horizontal_results if (r["x"] + r["width"] / 2.0) >= mid_split_x]

        has_two_sides = False
        if len(left_side) >= 3 and len(right_side) >= 3:
            left_right_edge = max(r["x"] + r["width"] for r in left_side)
            right_left_edge = min(r["x"] for r in right_side)
            center_gap = right_left_edge - left_right_edge
            if center_gap >= 15 or total_span_w > 450:
                has_two_sides = True

        column_partitions = []
        if has_two_sides:
            if total_span_w > 550:
                column_partitions.append(("左ページ本文", left_side))
                column_partitions.append(("右ページ本文", right_side))
            else:
                column_partitions.append(("左段本文", left_side))
                column_partitions.append(("右段本文", right_side))
        else:
            column_partitions.append(("本文横", horizontal_results))

        for col_name, col_items in column_partitions:
            if not col_items:
                continue

            col_items_sorted = sorted(col_items, key=lambda r: r["y"])
            median_h = np.median([r["height"] for r in col_items_sorted]) if col_items_sorted else 30
            line_gap = max(60, int(median_h * 2.2))

            groups = []
            current_group = []

            for r in col_items_sorted:
                if not current_group:
                    current_group.append(r)
                    continue

                previous = current_group[-1]
                gap = abs(r["y"] - previous["y"])

                if gap <= line_gap:
                    current_group.append(r)
                else:
                    groups.append(current_group)
                    current_group = [r]

            if current_group:
                groups.append(current_group)

            for g_idx, group in enumerate(groups):
                if not group:
                    continue

                valid_items = [r for r in group if r.get("text", "").strip()]
                if not valid_items:
                    valid_items = group

                min_x = min(r["x"] for r in valid_items)
                min_y = min(r["y"] for r in valid_items)
                max_x = max(r["x"] + r["width"] for r in valid_items)
                max_y = max(r["y"] + r["height"] for r in valid_items)

                display_name = col_name if len(groups) == 1 else f"{col_name}_{g_idx + 1}"

                regions.append(
                    {
                        "name": display_name,
                        "type": "body",
                        "x": int(min_x),
                        "y": int(min_y),
                        "width": int(max_x - min_x),
                        "height": int(max_y - min_y),
                        "orientation": "horizontal",
                        "ocr_count": len(valid_items)
                    }
                )

    return regions

# =========================================================
# メイン処理
# =========================================================

def main():

    # -----------------------------------------------------
    # 引数確認
    # -----------------------------------------------------

    import argparse
    parser = argparse.ArgumentParser(description="NDLOCR Auto Region Extractor")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("--orientation", choices=["auto", "vertical", "horizontal"], default="auto", help="Text orientation preference")
    parser.add_argument("--doc-type", choices=["japanese", "western"], default="japanese", help="Document type")

    args, unknown = parser.parse_known_args()

    image_path = Path(args.image_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    orientation_mode = args.orientation
    doc_type = args.doc_type

    # -----------------------------------------------------
    # 入力画像確認・見開き検知・OCR実行
    # -----------------------------------------------------

    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        print(f"画像がありません: {image_path}", file=sys.stderr)
        sys.exit(1)

    is_spread, split_x = is_spread_image(orig_img)
    venv_py = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        python_exe = venv_py
    else:
        python_exe = Path(sys.executable)

    if is_spread:
        print(f"[見開き検知] 2ページ見開き画像を検出 (W={orig_img.shape[1]}, H={orig_img.shape[0]}). 中央分割位置 X={split_x} で左右独立前処理・OCRを実行します", flush=True)

        left_raw = orig_img[:, :split_x]
        right_raw = orig_img[:, split_x:]

        left_out_dir = output_dir / "spread_left"
        right_out_dir = output_dir / "spread_right"
        left_out_dir.mkdir(parents=True, exist_ok=True)
        right_out_dir.mkdir(parents=True, exist_ok=True)

        left_prep_path = left_out_dir / "preprocessed_left.png"
        right_prep_path = right_out_dir / "preprocessed_right.png"

        _, left_info = preprocess_image_array(left_raw, left_prep_path, orientation_mode, doc_type)
        _, right_info = preprocess_image_array(right_raw, right_prep_path, orientation_mode, doc_type)

        def run_spread_ocr(prep_p, out_d):
            cmd = [
                str(python_exe), "-m", "ocr",
                "--sourceimg", str(prep_p),
                "--output", str(out_d),
                "--json-only",
                "--device", "cpu",
                "--det-score-threshold", "0.15",
                "--det-conf-threshold", "0.15"
            ]
            if orientation_mode == "vertical" and doc_type != "western":
                cmd.append("--enable-tcy")
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            if res.returncode != 0:
                print(f"[警告] OCR実行エラー ({prep_p.name}): {res.stderr}", file=sys.stderr, flush=True)
            j_path = find_json_file(out_d, prep_p)
            return j_path

        left_json = run_spread_ocr(left_prep_path, left_out_dir)
        right_json = run_spread_ocr(right_prep_path, right_out_dir)

        left_results = parse_ndlocr_json(left_json, left_info)
        right_results = parse_ndlocr_json(right_json, right_info)

        for r in right_results:
            r["x"] += split_x

        results = left_results + right_results
        ocr_target_path = image_path
        transform_info = None
        json_path = left_json

    else:
        ocr_target_path, transform_info = preprocess_image_for_ocr(
            image_path,
            output_dir,
            orientation_mode=orientation_mode,
            doc_type=doc_type
        )
        command = [
            str(python_exe),
            "-m",
            "ocr",
            "--sourceimg",
            str(ocr_target_path),
            "--output",
            str(output_dir),
            "--json-only",
            "--device",
            "cpu",
            "--det-score-threshold",
            "0.15",
            "--det-conf-threshold",
            "0.15"
        ]
        if orientation_mode == "vertical" and doc_type != "western":
            command.append("--enable-tcy")

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if process.returncode != 0:
            print(f"NDLOCR-Lite終了コード: {process.returncode}", file=sys.stderr, flush=True)
            sys.exit(process.returncode)

        json_path = find_json_file(output_dir, ocr_target_path)
        results = parse_ndlocr_json(json_path, transform_info)

    # -----------------------------------------------------
    # OCR結果表示
    # -----------------------------------------------------

    print()

    print(
        "========== OCR検出結果 =========="
    )

    for i, r in enumerate(results):

        print(
            f"[{i:02d}] "
            f"x={r['x']}, "
            f"y={r['y']}, "
            f"w={r['width']}, "
            f"h={r['height']}, "
            f"vertical={r['isVertical']}, "
            f"text={r['text']}"
        )

    print(
        "=================================="
    )

    print()

    # -----------------------------------------------------
    # 画像サイズ取得
    #
    # NDLOCRのimginfoが0でも、
    # 実際の画像から取得する
    # -----------------------------------------------------

    try:

        with Image.open(
            image_path
        ) as img:

            image_width, image_height = (
                img.size
            )

    except Exception as e:

        print(
            f"画像サイズ取得エラー: {e}",
            file=sys.stderr
        )

        image_width = 0
        image_height = 0

    print(
        f"実画像サイズ: "
        f"{image_width}×{image_height}",
        flush=True
    )

    # -----------------------------------------------------
    # C#用OCR JSON
    # -----------------------------------------------------

    result = {

        "image": str(
            image_path
        ),

        "json": str(
            json_path
        ),

        "results": results
    }

    output_json = (
        output_dir
        / "auto_regions.json"
    )

    with output_json.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    # page.json としても保存（C#互換性保証）
    page_json_file = output_dir / "page.json"
    with page_json_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"自動領域JSON: "
        f"{output_json}",
        flush=True
    )

    print(
        f"検出数: "
        f"{len(results)}",
        flush=True
    )

    # -----------------------------------------------------
    # 罫線検出テスト（前処理済み画像から高精度検出）
    # -----------------------------------------------------

    line_result = detect_lines(
        ocr_target_path,
        output_dir
    )

    # -----------------------------------------------------
    # 罫線座標解析
    # -----------------------------------------------------

    line_data = analyze_detected_lines(
        ocr_target_path,
        output_dir
    )

    # -----------------------------------------------------
    # 罫線領域生成
    # -----------------------------------------------------

    bordered_regions = create_bordered_regions_from_lines(
        line_data,
        image_width,
        image_height
    )

    # -----------------------------------------------------
    # 罫線領域内部の解析
    # -----------------------------------------------------

    analyzed_bordered_regions = analyze_bordered_region_lines(
        line_data,
        bordered_regions
    )

    # -----------------------------------------------------
    # セル候補生成
    # -----------------------------------------------------

    cell_candidate_regions = create_cell_candidates(
        analyzed_bordered_regions
    )

    # -----------------------------------------------------
    # セル候補へのOCR割り当て
    # -----------------------------------------------------

    analyzed_cell_regions = assign_ocr_to_cell_candidates(
        cell_candidate_regions,
        results
    )

    # -----------------------------------------------------
    # セルOCR結果から表領域を生成
    # -----------------------------------------------------

    table_regions = create_table_regions_from_cells(
        analyzed_cell_regions
    )

    if transform_info:
        for tbl in table_regions:
            rect = transform_rect_back(tbl["x"], tbl["y"], tbl["width"], tbl["height"], transform_info)
            tbl["x"] = rect["x"]
            tbl["y"] = rect["y"]
            tbl["width"] = rect["width"]
            tbl["height"] = rect["height"]

    print()
    print(
        "========== 表領域生成開始 ==========",
        flush=True
    )

    print(
        f"検出表数: {len(table_regions)}",
        flush=True
    )

    for table_index, table in enumerate(
        table_regions,
        start=1
    ):

        print(
            f"  表{table_index}: "
            f"x={table['x']}, "
            f"y={table['y']}, "
            f"width={table['width']}, "
            f"height={table['height']}, "
            f"rows={table['rows']}, "
            f"columns={table['columns']}",
            flush=True
        )

    print(
        "========== 表領域生成終了 ==========",
        flush=True
    )

    # -----------------------------------------------------
    # 表セル内のOCRを本文判定から除外
    # -----------------------------------------------------

    body_results = remove_table_ocr(
        results,
        analyzed_cell_regions
    )

    print()
    print(
        "========== 本文OCR候補作成 ==========",
        flush=True
    )

    print(
        f"全OCR数: {len(results)}",
        flush=True
    )

    print(
        f"表セル内OCR除外後: {len(body_results)}",
        flush=True
    )

    # -----------------------------------------------------
    # 表見出しを抽出
    #
    # 表の直上にあるOCRだけを表見出し候補とする。
    # 表の下にあるOCRは表へ取り込まず、本文候補として残す。
    # -----------------------------------------------------

    def extract_table_captions(
        body_results,
        table_regions
    ):
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

    table_captions = extract_table_captions(
        body_results,
        table_regions
    )

    print()
    print(
        "========== 表見出し判定 ==========",
        flush=True
    )

    caption_ids = set()

    for item in table_captions:
        table_index = item["table_index"]
        caption = item["ocr"]

        caption_ids.add(id(caption))

        print(
            f"表{table_index} 見出し候補: "
            f"x={caption['x']}, "
            f"y={caption['y']}, "
            f"text={caption['text']}",
            flush=True
        )

    # -----------------------------------------------------
    # 表見出しを除いた本文候補
    #
    # 表の下にある本文OCRはここに残る。
    # -----------------------------------------------------

    body_results_without_captions = [
        ocr
        for ocr in body_results
        if id(ocr) not in caption_ids
    ]

    print(
        f"見出し除外後本文候補数: "
        f"{len(body_results_without_captions)}",
        flush=True
    )

    # -----------------------------------------------------
    # 表以外のOCRから本文領域を生成
    # -----------------------------------------------------

    body_regions = create_body_regions(
        body_results_without_captions,
        orientation_mode=orientation_mode,
        doc_type=doc_type
    )

    print(
        f"本文領域数: {len(body_regions)}",
        flush=True
    )

    for region in body_regions:
        print(
            f"  {region['name']}: "
            f"x={region['x']}, "
            f"y={region['y']}, "
            f"width={region['width']}, "
            f"height={region['height']}, "
            f"orientation={region.get('orientation')}, "
            f"ocr_count={region.get('ocr_count')}",
            flush=True
        )

    print(
        "========== 本文OCR候補作成終了 ==========",
        flush=True
    )

    # -----------------------------------------------------
    # 表領域 + 本文領域
    # -----------------------------------------------------

    regions = (
        table_regions
        + body_regions
    )

    # -----------------------------------------------------
    # auto_layout.json
    # -----------------------------------------------------

    layout_result = {

        "image": str(
            image_path
        ),

        "image_width": image_width,

        "image_height": image_height,

        "regions": regions
    }

    layout_json = (

        output_dir
        / "auto_layout.json"
    )

    with layout_json.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            layout_result,

            f,

            ensure_ascii=False,

            indent=2
        )

    print(
        f"自動レイアウトJSON: "
        f"{layout_json}",
        flush=True
    )

    print(
        f"自動領域数: "
        f"{len(regions)}",
        flush=True
    )

    # -----------------------------------------------------
    # 領域表示
    # -----------------------------------------------------

    for region in regions:

        print(

            f"  {region['name']}: "

            f"x={region['x']}, "

            f"y={region['y']}, "

            f"width={region['width']}, "

            f"height={region['height']}, "

            f"type={region['type']}",

            flush=True
        )
    

# =========================================================
# プログラム開始
# =========================================================

if __name__ == "__main__":

    main()