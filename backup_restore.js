// Backups stay on this PC or its attached storage; no browser downloads.
let dataBackupMode = 'backup';
let dataBackupPreviewPath = '';
let dataBackupPolling = null;
const dataBackupElement = id => document.getElementById(id);
async function dataBackupApi(path, body) {
    const token = window.__APP_SECURITY_TOKEN__ || document.querySelector('meta[name="app-token"]')?.content || '';
    const response = await fetch('/api/backup/' + path, {
        method: body === undefined ? 'GET' : 'POST', cache: 'no-store',
        headers: {'Content-Type': 'application/json', 'X-App-Token': token},
        ...(body === undefined ? {} : {body: JSON.stringify(body)})
    });
    let data;
    try { data = await response.json(); } catch (_) { throw new Error('アプリに接続できません。起動状態を確認してください。'); }
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '処理を完了できませんでした。');
    return data;
}
function dataBackupSize(bytes) { return (bytes / (1024 ** 3)).toLocaleString('ja', {maximumFractionDigits: 2}) + ' GB'; }
function dataBackupMessage(text) { dataBackupElement('dataBackupStatus').textContent = text; }
async function openDataBackup(mode) {
    dataBackupMode = mode; dataBackupPreviewPath = '';
    const backup = mode === 'backup';
    dataBackupElement('dataBackupTitle').textContent = backup ? 'バックアップを作成' : 'バックアップから復元';
    dataBackupElement('dataBackupPathLabel').textContent = backup ? '保存先フォルダ' : '復元するバックアップフォルダ';
    dataBackupElement('dataBackupHelp').textContent = backup ? '指定した場所に、日付付きのバックアップフォルダを作成します。既存のバックアップは上書きしません。' : 'backup.jsonが入ったフォルダを選択してください。検証後、アプリの再起動時に現在のDB・PDF・図版を置き換えます。復元前のデータは自動で退避します。';
    dataBackupElement('dataBackupRun').textContent = backup ? 'バックアップを作成' : '内容を確認';
    dataBackupElement('dataBackupRun').disabled = false;
    dataBackupElement('dataBackupPath').disabled = false;
    dataBackupElement('dataBackupBrowse').disabled = false;
    dataBackupElement('dataBackupPreview').textContent = '';
    dataBackupElement('dataBackupConfirm').hidden = true;
    dataBackupElement('dataBackupConsent').checked = false;
    dataBackupElement('dataBackupCancelRestore').hidden = true;
    dataBackupElement('dataBackupProgress').hidden = true;
    dataBackupMessage('先に未保存の編集を保存し、OCR・検索・回答作成の完了を待ってください。');
    try { dataBackupElement('dataBackupPath').value = localStorage.getItem('data-backup-path-' + mode) || ''; } catch (_) { dataBackupElement('dataBackupPath').value = ''; }
    const dialog = dataBackupElement('dataBackupDialog');
    if (!dialog.open) dialog.showModal();
    await refreshDataBackup();
}
async function chooseDataBackupFolder() {
    const button = dataBackupElement('dataBackupBrowse'); button.disabled = true;
    dataBackupMessage('フォルダ選択画面を開いています。画面が隠れている場合はタスクバーも確認してください。');
    try {
        const result = await dataBackupApi('choose-folder', {purpose: dataBackupMode});
        if (result.path) { dataBackupElement('dataBackupPath').value = result.path; resetDataBackupPreview(); }
        dataBackupMessage(result.path ? 'フォルダを選択しました。' : 'フォルダ選択を取り消しました。');
    } catch (error) { dataBackupMessage(error.message); }
    finally { button.disabled = false; }
}
function resetDataBackupPreview() {
    dataBackupPreviewPath = ''; dataBackupElement('dataBackupConfirm').hidden = true;
    dataBackupElement('dataBackupConsent').checked = false; dataBackupElement('dataBackupPreview').textContent = '';
    dataBackupElement('dataBackupRun').textContent = dataBackupMode === 'backup' ? 'バックアップを作成' : '内容を確認';
}
async function runDataBackup() {
    const path = dataBackupElement('dataBackupPath').value.trim();
    if (!path) { dataBackupMessage('フォルダを選択するか、パスを入力してください。'); return; }
    const button = dataBackupElement('dataBackupRun'); button.disabled = true;
    try {
        if (dataBackupMode === 'restore' && dataBackupPreviewPath !== path) {
            const preview = await dataBackupApi('preview', {path});
            dataBackupPreviewPath = path;
            dataBackupElement('dataBackupPreview').textContent = `保存日時：${new Date(preview.created_at).toLocaleString('ja')}\nPDF：${preview.pdf_count.toLocaleString('ja')}件 ／ ファイル：${preview.file_count.toLocaleString('ja')}件 ／ ${dataBackupSize(preview.bytes)}`;
            dataBackupElement('dataBackupConfirm').hidden = false;
            button.textContent = '検証して復元を準備';
            dataBackupMessage('復元する内容を確認してください。反映までは現在のデータを変更しません。');
            return;
        }
        if (dataBackupMode === 'restore' && !dataBackupElement('dataBackupConsent').checked) {
            dataBackupMessage('現在のデータを置き換えることを確認し、確認欄にチェックしてください。'); return;
        }
        await dataBackupApi(dataBackupMode === 'backup' ? 'create' : 'restore', {path});
        try { localStorage.setItem('data-backup-path-' + dataBackupMode, path); } catch (_) {}
        dataBackupElement('dataBackupPath').disabled = true;
        dataBackupElement('dataBackupBrowse').disabled = true;
        dataBackupMessage('処理を開始しました。完了までアプリの保存・編集を一時停止します。');
        await refreshDataBackup();
    } catch (error) { dataBackupMessage(error.message); }
    finally { button.disabled = dataBackupElement('dataBackupPath').disabled; }
}
async function refreshDataBackup() {
    try {
        const result = await dataBackupApi('status');
        const location = dataBackupElement('dataBackupPath');
        if (dataBackupMode === 'backup' && location && !location.value && !location.disabled) location.value = result.default_backup_directory || '';
        const job = result.jobs[result.jobs.length - 1];
        const running = job?.state === 'running';
        dataBackupElement('dataBackupRun').disabled = running || result.pending;
        dataBackupElement('dataBackupPath').disabled = running || result.pending;
        dataBackupElement('dataBackupBrowse').disabled = running || result.pending;
        dataBackupElement('dataBackupCancelRestore').hidden = !result.pending;
        if (job) {
            dataBackupMessage(job.message + (job.state === 'completed' && job.path ? '\n保存先：' + job.path : ''));
            const progress = dataBackupElement('dataBackupProgress'); progress.hidden = !running;
            if (job.total_bytes) { progress.max = job.total_bytes; progress.value = Math.min(job.bytes, job.total_bytes); }
            else progress.removeAttribute('value');
            dataBackupElement('dataBackupProgressText').textContent = running && job.total_bytes ? `${dataBackupSize(job.bytes)} / ${dataBackupSize(job.total_bytes)}（検証完了までお待ちください）` : '';
        } else if (result.pending) {
            dataBackupMessage('復元の準備が完了しています。アプリを終了して再起動してください。反映まで保存・編集を一時停止しています。');
        } else if (result.last_restore) {
            dataBackupMessage(result.last_restore.message + (result.last_restore.history ? '\n復元前の退避先：' + result.last_restore.history : ''));
        }
        const menu = document.querySelector('[data-backup-menu-status]');
        if (menu) menu.textContent = result.pending ? '復元準備済み：アプリを終了して再起動してください。' : running ? 'バックアップ・復元の処理中です。' : result.last_restore?.message || '';
        if (dataBackupPolling) { clearTimeout(dataBackupPolling); dataBackupPolling = null; }
        if (running) dataBackupPolling = setTimeout(refreshDataBackup, 1200);
    } catch (error) {
        dataBackupMessage(error.message);
        if (dataBackupPolling || dataBackupElement('dataBackupPath').disabled) dataBackupPolling = setTimeout(refreshDataBackup, 3000);
    }
}
async function cancelDataRestore() {
    const button = dataBackupElement('dataBackupCancelRestore'); button.disabled = true;
    try { await dataBackupApi('cancel-restore', {}); await refreshDataBackup(); }
    catch (error) { dataBackupMessage(error.message); }
    finally { button.disabled = false; }
}
window.addEventListener('DOMContentLoaded', () => {
    dataBackupElement('dataBackupPath').addEventListener('input', resetDataBackupPreview);
    refreshDataBackup();
});
