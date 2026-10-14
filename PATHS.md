# 保存先・パスの管理

## 設定する場所

- アプリ内部の共通定義: app/paths.py の APP_PATHS。
- 外部バックアップの既定先: config/paths.json の backup_directory。

本体の位置はapp/paths.py自身の位置から求めます。ユーザー名や配置先をコードに書き込みません。data、モデル、OCRエンジン、リサーチ環境、各DB、辞書、画像などの標準保存先は共通定義から取得します。

外部バックアップの設定例:

    {"backup_directory": "~/BackUp/OCRアプリ"}

~ は現在のWindowsユーザーのホームです。別ドライブではD:/MyNotebookLM_Backupsなどを指定できます。相対指定は本体を基準に解決します。バックアップ画面はこの値を既定先に使い、画面で明示的に選んだ保存先の記憶があればそちらを優先します。

## DBと読み書き

内部参照はdata/imports/document.docxなど、本体からの相対パスとして保存します。外部の原本ファイルや外部URLは保持します。画像のブラウザ用URL /media/... もユーザー名に依存しません。

保存時はAPP_PATHS.store_file、ファイルを開く直前はAPP_PATHS.resolve_file、図版はAPP_PATHS.media_file / media_urlを使います。本文や注釈に書かれたパス文字列は書き換えません。既知のメタデータの参照と、HTMLの画像・リンク属性のみを対象にします。

## フォルダーを移すとき

本体をdata、config、models、runtimes、.venv_researchごと移動すれば、共通定義と相対参照は新しい場所を自動認識します。内部パスをDBへ再設定する必要はありません。専用Python環境の修復と、Windowsの起動登録・自動整理の入口などの外部登録先は移行手順に従って更新します。

古い絶対パスを含むバックアップを復元する場合の確認:

    py -3.14 -m app.path_migration --check

必要な場合だけ、変更前DBのバックアップを自動作成して適用:

    py -3.14 -m app.path_migration --apply

別の旧配置を明示する場合は--source-rootに旧アプリ本体の場所を指定します。外部原本を推測で移し替える処理は行いません。

## バックアップの自動整理

自動整理もAPP_PATHS.backup_directoryとbackup_logを参照します。専用フォルダーの.mynotebooklm-backup-root.jsonを確認し、識別情報がない場所では削除しません。識別ファイル自体は期限切れ削除から除外します。保存先を移す場合は、専用フォルダーを識別情報ごと移し、config/paths.jsonを更新します。

既存の保存期限と通知条件は維持しています。成果物や原本文書、本番DBを期限切れ整理の対象にはしません。
