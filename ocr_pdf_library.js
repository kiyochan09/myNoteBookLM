// Stored PDF editing stays in the app. Drafts live in the current tabs until
// the user explicitly saves a new verified PDF version.
let currentArchiveEditor = null;
let ocrPdfLibraryDialog = null;
let ocrPdfLibraryRequest = null;
let ocrPdfLibrarySequence = 0;
let ocrPdfLibraryTimer = null;
const ocrPdfNormalControlStates = new Map();

function updateOcrPdfEditorState() {
    const editor = currentArchiveEditor;
    const status = document.getElementById('ocrPdfEditorStatus');
    if (status) {
        status.hidden = !editor;
        status.textContent = editor ? `${editor.title} — ${editor.dirty ? '変更あり・PDFへ未保存' : 'PDF保存済み'}` : '';
    }
    const save = document.getElementById('btnSaveEditedOcrPdf');
    if (save) { save.hidden = !editor; save.disabled = !editor?.dirty || ocrPdfArchiveBusy; }
    const persistence = document.getElementById('lblBatchPersistence');
    if (persistence) persistence.textContent = editor ? '編集内容は「変更をPDFへ保存」で反映' : 'OCR結果・ユーザー補正をリアルタイム記録';
    for (const id of ['btnSaveOcrPdf', 'btnDeleteDiskOcrData', 'btnStartOcr', 'btnDetectRegions', 'btnImageProcessing', 'btnSaveAndCloseBatch', 'btnResetUiCache']) {
        const control = document.getElementById(id);
        if (!control) continue;
        if (editor) {
            if (!ocrPdfNormalControlStates.has(id)) ocrPdfNormalControlStates.set(id, control.disabled);
            control.disabled = true;
        } else if (ocrPdfNormalControlStates.has(id)) {
            control.disabled = ocrPdfNormalControlStates.get(id);
            ocrPdfNormalControlStates.delete(id);
        }
    }
}

function setOcrPdfEditorDirty() {
    if (currentArchiveEditor && !currentArchiveEditor.loading) {
        currentArchiveEditor.dirty = true;
        updateOcrPdfEditorState();
    }
}

function confirmOcrPdfEditorSwitch() {
    return !currentArchiveEditor?.dirty || confirm('PDFへ未保存の変更があります。変更を破棄して別のPDFを開きますか？\n\n残す場合はキャンセルし、「変更をPDFへ保存」を押してください。');
}

function ensureOcrPdfLibrary() {
    if (ocrPdfLibraryDialog) return ocrPdfLibraryDialog;
    const dialog = document.createElement('dialog');
    dialog.id = 'ocrPdfLibraryDialog';
    dialog.className = 'ocr-pdf-library';
    dialog.setAttribute('aria-labelledby', 'ocrPdfLibraryTitle');
    dialog.innerHTML = `<div class="ocr-pdf-library-heading"><strong id="ocrPdfLibraryTitle">透明テキスト編集 — 保存済みPDF</strong><button type="button" id="ocrPdfLibraryClose" class="win-btn">閉じる</button></div>
        <form id="ocrPdfLibrarySearch" class="ocr-pdf-library-tools">
            <input id="ocrPdfLibraryQuery" type="search" placeholder="文書名を検索" aria-label="検索語">
            <label><input id="ocrPdfLibraryIncludeText" type="checkbox"> 本文も検索</label>
            <select id="ocrPdfLibrarySort" aria-label="並べ替え"><option value="saved-desc">保存日時：新しい順</option><option value="saved-asc">保存日時：古い順</option><option value="name-asc">文書名：昇順</option><option value="name-desc">文書名：降順</option></select>
            <button type="submit" class="win-btn">検索・更新</button>
        </form>
        <p id="ocrPdfLibraryMessage" role="status" aria-live="polite"></p>
        <div class="ocr-pdf-library-scroll"><table><thead><tr><th>文書名</th><th>保存日時</th><th>ページ</th><th>操作</th></tr></thead><tbody id="ocrPdfLibraryRows"></tbody></table></div>
        <p class="ocr-pdf-library-help">アプリ内のPDFを直接開き、編集後は同じファイルへ上書き保存します。</p>`;
    document.body.appendChild(dialog);
    const refresh = () => { clearTimeout(ocrPdfLibraryTimer); loadOcrPdfLibrary(); };
    dialog.querySelector('#ocrPdfLibrarySearch').onsubmit = event => { event.preventDefault(); refresh(); };
    dialog.querySelector('#ocrPdfLibraryQuery').oninput = () => {
        clearTimeout(ocrPdfLibraryTimer);
        // Invalidate older results immediately, including during the debounce.
        ocrPdfLibrarySequence++;
        ocrPdfLibraryRequest?.abort();
        ocrPdfLibraryTimer = setTimeout(loadOcrPdfLibrary, 350);
    };
    dialog.querySelector('#ocrPdfLibrarySort').onchange = refresh;
    dialog.querySelector('#ocrPdfLibraryIncludeText').onchange = () => {
        dialog.querySelector('#ocrPdfLibraryQuery').placeholder = dialog.querySelector('#ocrPdfLibraryIncludeText').checked ? '文書名・PDF本文を検索' : '文書名を検索';
        refresh();
    };
    dialog.querySelector('#ocrPdfLibraryClose').onclick = () => { if (!ocrPdfArchiveBusy) dialog.close(); };
    dialog.addEventListener('cancel', event => { if (ocrPdfArchiveBusy) event.preventDefault(); });
    dialog.addEventListener('close', () => { clearTimeout(ocrPdfLibraryTimer); ocrPdfLibraryRequest?.abort(); });
    ocrPdfLibraryDialog = dialog;
    return dialog;
}

async function openOcrPdfLibrary() {
    if (ocrPdfArchiveBusy || isAutoLayoutRunning) { alert('現在の処理が完了してから実行してください。'); return; }
    const dialog = ensureOcrPdfLibrary();
    if (!dialog.open) dialog.showModal();
    await loadOcrPdfLibrary();
    dialog.querySelector('#ocrPdfLibraryQuery').focus();
}

async function loadOcrPdfLibrary() {
    const dialog = ensureOcrPdfLibrary();
    if (ocrPdfArchiveBusy) return;
    ocrPdfLibraryRequest?.abort();
    const controller = new AbortController(); ocrPdfLibraryRequest = controller;
    const sequence = ++ocrPdfLibrarySequence;
    const [sort, direction] = dialog.querySelector('#ocrPdfLibrarySort').value.split('-');
    const params = new URLSearchParams({q:dialog.querySelector('#ocrPdfLibraryQuery').value,
        include_text:String(dialog.querySelector('#ocrPdfLibraryIncludeText').checked), sort, descending:String(direction === 'desc')});
    const message = dialog.querySelector('#ocrPdfLibraryMessage');
    const rows = dialog.querySelector('#ocrPdfLibraryRows');
    message.textContent = '保存済みPDFを確認しています…'; rows.replaceChildren();
    try {
        const response = await fetch(`${API_BASE}/api/batch/ocr-pdfs?${params}`, {signal:controller.signal, cache:'no-store'});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '一覧を読み込めません。');
        if (sequence !== ocrPdfLibrarySequence || !dialog.open) return;
        for (const item of data.items || []) {
            const row = document.createElement('tr');
            const name = document.createElement('td');
            const title = document.createElement('strong'); title.textContent = item.document_name; name.appendChild(title);
            if (item.folder_name !== item.document_name) { const folder = document.createElement('small'); folder.textContent = item.folder_name; name.appendChild(folder); }
            const date = document.createElement('td'); date.textContent = new Date(item.saved_at).toLocaleString('ja-JP');
            const pages = document.createElement('td'); pages.textContent = `${item.text_page_count} / ${item.total_pages}`;
            const action = document.createElement('td');
            const open = document.createElement('button'); open.type = 'button'; open.className = 'win-btn win-btn-primary'; open.textContent = '開いて編集';
            open.onclick = () => openStoredOcrPdf(item.archive_id); action.appendChild(open);
            row.append(name, date, pages, action); rows.appendChild(row);
        }
        message.textContent = data.items?.length ? `${data.items.length}件のPDF` : '該当する保存済みPDFがありません。';
    } catch (error) {
        if (error.name !== 'AbortError' && sequence === ocrPdfLibrarySequence) message.textContent = '読み込みに失敗しました。' + error.message;
    }
}

async function openStoredOcrPdf(archiveId) {
    if (ocrPdfArchiveBusy || isAutoLayoutRunning || !confirmOcrPdfEditorSwitch()) return false;
    const dialog = ensureOcrPdfLibrary();
    const message = dialog.querySelector('#ocrPdfLibraryMessage');
    message.textContent = 'PDF画像と編集データを読み込んでいます…';
    ocrPdfArchiveBusy = true;
    try {
        await flushPendingPageSaves();
        const response = await fetch(`${API_BASE}/api/batch/ocr-pdfs/${encodeURIComponent(archiveId)}/edit-data`, {cache:'no-store'});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '保存済みPDFを開けません。');
        const editor = {id:archiveId, title:data.document_name, filename:data.output_name, dirty:false, loading:true};
        const loaded = await handleFileSelect(new File([], data.output_name, {type:'application/pdf'}),
            {confirmed:true, editor, renderedPage:data.rendered_page, portablePages:data.pages});
        if (!loaded) throw new Error('保存済みPDFを画面へ読み込めませんでした。');
        editor.loading = false; editor.dirty = false;
        modifiedPages.clear();
        for (const entry of Object.values(pageCache)) entry.isModified = false;
        dialog.close(); switchTab('body');
        appendLog('透明テキスト編集', `${data.document_name}を開きました。右側のタブを編集し、「変更をPDFへ保存」で反映できます。`, 'success');
        return true;
    } catch (error) {
        message.textContent = error.message;
        if (!dialog.open) alert(error.message);
        return false;
    } finally { ocrPdfArchiveBusy = false; updateOcrPdfEditorState(); }
}

async function saveEditedOcrPdf() {
    if (!currentArchiveEditor || !currentArchiveEditor.dirty || ocrPdfArchiveBusy) return;
    const editor = currentArchiveEditor;
    const overlay = document.createElement('dialog'); overlay.className = 'ocr-pdf-saving';
    overlay.textContent = '編集内容をPDFへ保存・検証しています…';
    overlay.addEventListener('cancel', event => event.preventDefault());
    document.body.appendChild(overlay); overlay.showModal();
    ocrPdfArchiveBusy = true; updateOcrPdfEditorState();
    try {
        await flushPendingPageSaves(); saveCurrentPageToCache();
        const pages = {};
        for (const [key, entry] of Object.entries(pageCache)) if (entry.docData) pages[key] = entry.docData;
        const response = await fetch(`${API_BASE}/api/batch/ocr-pdfs/${encodeURIComponent(editor.id)}/edit`,
            {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({pages})});
        const result = await response.json();
        if (!response.ok || result.status !== 'success') throw new Error(result.detail || 'PDFへ保存できませんでした。');
        editor.id = result.archive_id; editor.dirty = false;
        modifiedPages.clear();
        for (const entry of Object.values(pageCache)) entry.isModified = false;
        showStoredOcrPdfResult(result, false);
        appendLog('透明テキスト保存', `同じPDFへ上書き保存しました。${result.saved_path}`, 'success');
    } catch (error) {
        editor.dirty = true;
        alert('PDFへ保存できませんでした。編集内容は画面に残っています。\n' + error.message);
    } finally { ocrPdfArchiveBusy = false; overlay.close(); overlay.remove(); updateOcrPdfEditorState(); }
}

function showStoredOcrPdfResult(result, deleted) {
    const panel = document.getElementById('ocrPdfArchiveResult'); panel.replaceChildren(); panel.style.display = 'block';
    const message = document.createElement('div');
    const operation = {created:'透明テキストを作成・埋め込みました', updated:'透明テキストを更新しました', unchanged:'変更がないため埋め込みをスキップしました'}[result.operation] || 'PDFを保存しました';
    message.textContent = `${operation}（OCR ${result.page_count} / 全${result.total_pages}ページ）。保存先：${result.saved_path}${deleted ? ' 生データは削除済みです。' : ''}`;
    const open = document.createElement('button'); open.type = 'button'; open.className = 'batch-action-btn'; open.textContent = 'このPDFを開いて編集';
    open.onclick = () => openStoredOcrPdf(result.archive_id);
    panel.append(message, open);
    return message;
}

window.addEventListener('beforeunload', event => {
    if (currentArchiveEditor?.dirty) { event.preventDefault(); event.returnValue = ''; }
});

document.addEventListener('click', event => {
    if (event.target.closest('#pnlTableToolbar button')) setOcrPdfEditorDirty();
}, true);
