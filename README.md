# myNoteBookLM (MyNotebookLM & OCR Knowledge Base System)

高精度OCRレイアウト解析・縦中横照合エンジンおよびナレッジベース統合ドキュメントエディタシステム。

---

## 🌟 主な機能

1. **PDF / 書籍の高度なレイアウト解析 & OCR**
   - **NDLOCR & Windows OCR ハイブリッドエンジン**: 本文・見出し・表・注釈文・図版の自動区分検出。
   - **縦中横（TCY）画像照合エンジン**: 2桁数字（10〜99）の脱落や誤読を原本画像パッチ照合（NCC）により自動検出・復元。
   - **熟語・複合語誤爆防止ガード**: 文脈解析により自然な日本語の整合性を維持。

2. **対話型レイアウトデザイナー (Form1 Designer)**
   - **領域区分・属性編集**: 本文 (body)、表 (table)、見出し (heading)、注釈文 (footnote)、図版 (image) の右クリックコンテキストメニューによる直感的な種別変更。
   - **表罫線（Rule Lines）対話編集**: 表領域内の縦横罫線の追加・削除・位置調整およびセルマトリクス抽出。
   - **読み順（Reading Order）制御**: 段落・コラムに応じた読み順の番号付けと自動整流化。

3. **ドキュメントエクスポート & ナレッジベース管理**
   - **Word (.docx) ページ範囲指定出力**: ページまたぎ段落結合、ハイフン復元、ページ区切り制御。
   - **SQLite ナレッジベース**: 抽出されたテキスト・構造化データの全文検索および管理。

---

## 📂 ディレクトリ構成

`	ext
knowledge_base_system/
├── app/
│   ├── api/              # APIルーター・エンドポイント
│   ├── core/             # DB接続・スキーマ管理
│   ├── importers/        # PDF, DOCX, ODT インポーター
│   ├── ocr_pipeline/     # OCRエンジン、レイアウト解析、縦中横画像照合
│   │   ├── ndlocr_core/  # NDLOCR 抽出コア
│   │   ├── tcy_digit_refiner.py  # 縦中横照合・文字補正エンジン
│   │   └── layout_engine.py      # レイアウト構造化エンジン
│   ├── rag/              # ノートブックエンジン・検索
│   ├── db_service.py     # データベース操作サービス
│   ├── export_service.py # Word/Markdownエクスポート
│   └── main.py           # FastAPI アプリケーションエントリーポイント
├── data/
│   ├── ocr_results/      # OCR解析結果データ (JSON / Text)
│   └── tcy_templates/    # 縦中横マッチング用テンプレート画像
├── editor_form.html      # 統合ドキュメントエディタ UI
├── ocr_form1_designer.html # 対話型OCR領域デザイナー UI
├── ocr_test_full.html    # 全機能検証用テストUI
├── run.py                # アプリケーション起動スクリプト
└── requirements.txt      # 依存ライブラリ一覧
`

---

## 🚀 セットアップと起動方法

### 1. 動作要件
- Python 3.10 以上
- Windows 10 / 11

### 2. 依存ライブラリのインストール
`ash
pip install -r requirements.txt
`

### 3. サーバー起動
`ash
python run.py
`
起動後、ブラウザで以下のURLにアクセスしてください：
- **OCR デザイナー UI**: [http://localhost:8000/ocr_form1_designer.html](http://localhost:8000/ocr_form1_designer.html)
- **エディタ UI**: [http://localhost:8000/editor_form.html](http://localhost:8000/editor_form.html)
- **API ドキュメント**: [http://localhost:8000/docs](http://localhost:8000/docs)
