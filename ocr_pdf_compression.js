let pdfCompressionDialog = null;
let pdfCompressionSelection = null;
let pdfCompressionJob = null;
let pdfCompressionBusy = false;

function pdfCompressionSize(bytes) {
    return (bytes / 1000000).toLocaleString('ja-JP', {maximumFractionDigits: 2}) + ' MB';
}

async function pdfCompressionApi(path, body, method = 'POST') {
    const response = await fetch(`${API_BASE}/api/batch/pdf-compression${path}`, {
        method, cache: 'no-store', headers: {'Content-Type': 'application/json'},
        ...(body === undefined ? {} : {body: JSON.stringify(body)})
    });
    const data = await response.json();
    if (!response.ok) {
        const error = new Error(data.detail || 'PDF圧縮の処理に失敗しました。');
        error.status = response.status; throw error;
    }
    return data;
}

function pdfCompressionMessage(text) {
    document.getElementById('pdfCompressionStatus').textContent = text;
}

function pdfCompressionSetBusy(busy) {
    pdfCompressionBusy = busy;
    ocrPdfArchiveBusy = busy;
    document.getElementById('pdfCompressionSelect').disabled = busy;
    document.getElementById('pdfCompressionRun').disabled = busy || !pdfCompressionSelection;
    document.getElementById('pdfCompressionClose').disabled = busy;
    document.getElementById('pdfCompressionCancel').hidden = !pdfCompressionJob;
    if (typeof updateOcrPdfEditorState === 'function') updateOcrPdfEditorState();
}

function openPdfCompression() {
    if (ocrPdfArchiveBusy || isAutoLayoutRunning) { alert('現在の処理が終わってから実行してください。'); return; }
    if (currentArchiveEditor?.dirty) { alert('先に「変更をPDFへ保存」で編集内容を保存してください。'); return; }
    if (!pdfCompressionDialog) {
        const dialog = document.createElement('dialog');
        dialog.id = 'pdfCompressionDialog'; dialog.className = 'ocr-pdf-library';
        dialog.style.width = 'min(680px,94vw)';
        dialog.setAttribute('aria-labelledby', 'pdfCompressionTitle');
        dialog.innerHTML = `<div class="ocr-pdf-library-heading"><strong id="pdfCompressionTitle">PDF圧縮 — 透明テキスト・編集情報を保持</strong><button id="pdfCompressionClose" class="win-btn" type="button">閉じる</button></div>
            <p style="font-size:13px;line-height:1.8">フォルダーからPDFを選び、画像だけを容量優先で圧縮します。透明テキストと再編集用の本文・座標・読み順は保持します。</p>
            <button id="pdfCompressionSelect" class="win-btn" type="button">📂 フォルダーからPDFを選択…</button>
            <div id="pdfCompressionFile" style="margin:16px 0;padding:12px;background:#f1f5f9;border-radius:5px;overflow-wrap:anywhere;white-space:pre-line">PDFが選択されていません。</div>
            <p style="font-size:12px;line-height:1.8;color:#475569">保存先は同じフォルダーです。元のPDFは末尾に「old」を付けて退避し、圧縮PDFに元のファイル名を引き継ぎます。<br>例：書類.pdf → 元ファイル：書類old.pdf ／ 圧縮後：書類.pdf<br>同名のoldファイルがある場合は上書きしません。管理情報も更新・退避します。</p>
            <p style="font-size:12px;color:#64748b">画像の画質は低下します。JPEG画像は96 dpi、白黒画像は150 dpiを目安に、元より小さくなる画像を採用します。</p>
            <progress id="pdfCompressionProgress" hidden max="100" value="0" style="width:100%"></progress>
            <p id="pdfCompressionStatus" role="status" aria-live="polite" style="font-size:13px;line-height:1.8;white-space:pre-line;overflow-wrap:anywhere"></p>
            <div style="display:flex;gap:12px;justify-content:flex-end"><button id="pdfCompressionCancel" class="win-btn" type="button" hidden>キャンセル</button><button id="pdfCompressionRun" class="win-btn win-btn-primary" type="button" disabled>圧縮して同じフォルダーへ保存</button></div>`;
        document.body.appendChild(dialog);
        dialog.querySelector('#pdfCompressionClose').onclick = () => dialog.close();
        dialog.querySelector('#pdfCompressionSelect').onclick = choosePdfCompression;
        dialog.querySelector('#pdfCompressionRun').onclick = runPdfCompression;
        dialog.querySelector('#pdfCompressionCancel').onclick = cancelPdfCompression;
        dialog.addEventListener('cancel', event => { if (pdfCompressionBusy) event.preventDefault(); });
        pdfCompressionDialog = dialog;
    }
    if (!pdfCompressionDialog.open) pdfCompressionDialog.showModal();
}

async function choosePdfCompression() {
    if (pdfCompressionBusy) return;
    pdfCompressionSelection = null; pdfCompressionSetBusy(true);
    pdfCompressionMessage('ファイル選択画面を開いています。フォルダーを開き、圧縮するPDFを選択してください。');
    document.getElementById('pdfCompressionFile').textContent = 'PDFが選択されていません。';
    try {
        const data = await pdfCompressionApi('/select');
        if (data.cancelled) { pdfCompressionMessage('選択をキャンセルしました。'); return; }
        pdfCompressionSelection = data.selection;
        document.getElementById('pdfCompressionFile').textContent = `${data.path}\n現在の容量：${pdfCompressionSize(data.size_bytes)}\n元ファイルの退避名：${data.backup_name}`;
        pdfCompressionMessage('選択したPDFを確認し、「圧縮して同じフォルダーへ保存」を押してください。');
    } catch (error) { pdfCompressionMessage(error.message); }
    finally { pdfCompressionSetBusy(false); }
}

async function runPdfCompression() {
    if (pdfCompressionBusy || !pdfCompressionSelection) return;
    pdfCompressionSetBusy(true); pdfCompressionMessage('圧縮を開始しています…');
    try {
        const data = await pdfCompressionApi('/start', {selection: pdfCompressionSelection});
        pdfCompressionJob = data.job_id; pdfCompressionSelection = null;
        document.getElementById('pdfCompressionProgress').hidden = false;
        document.getElementById('pdfCompressionProgress').value = 0;
        pdfCompressionSetBusy(true);
        pollPdfCompression();
    } catch (error) { pdfCompressionMessage(error.message); pdfCompressionSetBusy(false); }
}

async function pollPdfCompression() {
    if (!pdfCompressionJob) return;
    try {
        const data = await pdfCompressionApi('/' + encodeURIComponent(pdfCompressionJob), undefined, 'GET');
        document.getElementById('pdfCompressionProgress').value = data.progress || 0;
        document.getElementById('pdfCompressionCancel').disabled = !!data.committing || !!data.cancelled;
        pdfCompressionMessage(data.message);
        if (data.state === 'running') { setTimeout(pollPdfCompression, 1000); return; }
        if (data.state === 'completed' && data.result?.changed) {
            const result = data.result;
            pdfCompressionMessage(`${data.message}\n${pdfCompressionSize(result.before_bytes)} → ${pdfCompressionSize(result.after_bytes)}（${result.reduction_percent}%削減）\n圧縮PDF：${result.saved_path}\n元ファイル：${result.backup_path}`);
            appendLog('PDF圧縮', `${result.saved_path}（${result.reduction_percent}%削減・透明テキストと編集情報を保持）`, 'success');
        }
        pdfCompressionJob = null;
        document.getElementById('pdfCompressionProgress').hidden = true;
        document.getElementById('pdfCompressionCancel').disabled = false;
        pdfCompressionSetBusy(false);
    } catch (error) {
        if (error.status === 400 || error.status === 404) {
            pdfCompressionJob = null;
            document.getElementById('pdfCompressionProgress').hidden = true;
            pdfCompressionMessage(error.message); pdfCompressionSetBusy(false); return;
        }
        pdfCompressionMessage('進捗を確認できません。接続を再確認しています…\n' + error.message);
        setTimeout(pollPdfCompression, 3000);
    }
}

async function cancelPdfCompression() {
    if (!pdfCompressionJob) return;
    document.getElementById('pdfCompressionCancel').disabled = true;
    try {
        await pdfCompressionApi('/' + encodeURIComponent(pdfCompressionJob) + '/cancel');
        pdfCompressionMessage('現在の処理が終了したところでキャンセルします…');
    } catch (error) { pdfCompressionMessage(error.message); document.getElementById('pdfCompressionCancel').disabled = false; }
}

window.addEventListener('beforeunload', event => {
    if (pdfCompressionBusy) { event.preventDefault(); event.returnValue = ''; }
});
