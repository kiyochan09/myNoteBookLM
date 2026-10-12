// Save locations come from the running app; never retain another PC's paths.
async function refreshStorageFolderMenu() {
    if (typeof refreshDataBackup === 'function') refreshDataBackup();
    const menu = document.querySelector('[data-storage-folder-menu]');
    if (!menu) return;
    const status = menu.querySelector('[data-storage-status]');
    status.textContent = '保存先を確認しています…';
    try {
        const response = await fetch('/api/batch/storage-folders', {cache: 'no-store'});
        const data = await response.json();
        if (!response.ok || data.status !== 'success') throw new Error(data.detail || '保存先を取得できませんでした。');
        for (const item of data.items) {
            const path = menu.querySelector(`[data-storage-path="${item.id}"]`);
            if (path) path.textContent = item.path;
        }
        status.textContent = '';
    } catch (error) {
        status.textContent = error.message || '保存先を取得できませんでした。';
    }
}

async function openAppStorageFolder(kind, button) {
    if (button.disabled) return;
    const menu = button.closest('[data-storage-folder-menu]');
    const status = menu.querySelector('[data-storage-status]');
    button.disabled = true;
    status.textContent = 'フォルダを開いています…';
    try {
        const response = await fetch(`/api/batch/storage-folders/${encodeURIComponent(kind)}/open`, {method: 'POST'});
        const data = await response.json();
        if (!response.ok || data.status !== 'success') throw new Error(data.detail || 'フォルダを開けませんでした。');
        const path = menu.querySelector(`[data-storage-path="${data.id}"]`);
        if (path) path.textContent = data.path;
        status.textContent = `${data.label}の保存フォルダを開きました。`;
    } catch (error) {
        status.textContent = error.message || 'フォルダを開けませんでした。';
    } finally {
        button.disabled = false;
    }
}
