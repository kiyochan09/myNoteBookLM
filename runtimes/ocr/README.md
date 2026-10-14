# MyNotebookLM OCR

このフォルダーが、MyNotebookLMのOCRエンジンの唯一の配置先です。別アプリのOCR-REPOSには依存しません。

- `src/`：実際に利用する修正版OCRソース。認識・末尾欠落補完などの修正はここで管理します。
- `src/model/`：検出・文字認識のONNXモデル4個。ハッシュは `model-manifest.json` に記録。
- `src/config/`：認識文字と領域分類の定義。
- `venv/`：Python 3.12の専用実行環境と依存ライブラリ。`mynotebooklm_ocr.pth` で相対位置の `src` を読み込みます。
- `dependencies.lock.txt`：移行時の依存ライブラリのバージョン記録。
- `licenses/`：元のNDLOCR-Liteと依存コードのライセンス表示。

アプリ側の領域解析・表認識・画像補正は `app/ocr_pipeline/` が担い、共通の `engine_paths.py` からこのOCRを呼び出します。OCRを別系統に複製しません。

実行環境の修復はアプリ本体で `py -3.12 repair_runtime.py --ocr` を実行します。依存ライブラリやモデルを削除せず、Python設定とランチャーを現在の配置に合わせます。`ndlocr-lite` の再インストールで修正版ソースを置き換えないでください。

例：`venv\Scripts\python.exe -m ocr --help`。pipは `venv\Scripts\python.exe -m pip` から利用できます。
