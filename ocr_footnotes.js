// Note identity remains stable; the displayed number follows the body order.
let ocrNoteCatalog = {}, ocrNoteFile = '', ocrNotePending = null, ocrNoteArmed = false;
let ocrDisplayedNotes = [];
let ocrNoteCatalogLoaded = '', ocrNoteCatalogRequest = null;
async function ocrLoadNoteCatalog() {
    const filename=currentPdfFile?.name;if(!filename)return;
    if(ocrNoteCatalogLoaded===filename)return;
    if(ocrNoteCatalogRequest)return ocrNoteCatalogRequest;
    ocrNoteCatalogRequest=(async()=>{
        const response=await fetch(`${API_BASE}/api/ocr/footnote-catalog?filename=${encodeURIComponent(filename)}`);
        if(!response.ok)throw new Error('catalog');
        const data=await response.json();
        if(currentPdfFile?.name===filename){ocrNoteFile=filename;ocrNoteCatalog=data.pages || {};ocrNoteCatalogLoaded=filename;ocrRefreshNoteNumbers();}
    })();
    try{await ocrNoteCatalogRequest;}finally{ocrNoteCatalogRequest=null;}
}
function ocrNoteKey(note,index) { return typeof note === 'object' ? String(note.note_id || note.region_id || note.id || `note-${index}`) : `note-${index}`; }
function ocrNoteDoc(page) { return page === currentPage ? currentDocData : pageCache[page]?.docData; }
function ocrNotePages() {
    if (ocrNoteFile !== (currentPdfFile?.name || '')) { ocrNoteCatalog = {}; ocrNoteFile = currentPdfFile?.name || ''; }
    const pages = {...ocrNoteCatalog};
    for (const [page, cache] of Object.entries(pageCache)) if (cache?.docData) pages[page] = cache.docData;
    if (currentDocData) pages[currentPage] = currentDocData;
    return pages;
}
function ocrNoteReferences() {
    const result = new Map(); let number = 0;
    for (const [page,doc] of Object.entries(ocrNotePages()).sort((a,b)=>Number(a[0])-Number(b[0]))) {
        for (const match of (doc.body_text || '').matchAll(/⟦注:([a-zA-Z0-9_-]+)⟧/g)) {
            const link = (doc.footnote_links || []).find(l=>l.id === match[1]);
            result.set(match[1], {number:++number, page:Number(page), link});
        }
    }
    return result;
}
function ocrReadBodyText() {
    const box=document.getElementById('richTextBox1'); if (!box) return '';
    const nodes=[...box.querySelectorAll('.ocr-note-reference')];
    const values=nodes.map(n=>n.textContent);
    nodes.forEach(n=>n.textContent=`⟦注:${n.dataset.noteRef}⟧`);
    const text=box.innerText;
    nodes.forEach((n,i)=>n.textContent=values[i]);
    return text;
}
function ocrRefreshNoteNumbers() {
    const refs=ocrNoteReferences(),box=document.getElementById('richTextBox1');
    if (box) {
        // Render persisted tokens even when this is the first page opened.
        const walker=document.createTreeWalker(box,NodeFilter.SHOW_TEXT);const nodes=[];
        while(walker.nextNode()) if (/⟦注:[a-zA-Z0-9_-]+⟧/.test(walker.currentNode.nodeValue) && !walker.currentNode.parentElement.closest('.ocr-note-reference')) nodes.push(walker.currentNode);
        for(const node of nodes) {
            const fragment=document.createDocumentFragment();let start=0;
            for(const match of node.nodeValue.matchAll(/⟦注:([a-zA-Z0-9_-]+)⟧/g)) {
                fragment.append(document.createTextNode(node.nodeValue.slice(start,match.index)));
                const sup=document.createElement('sup');sup.className='ocr-note-reference';sup.dataset.noteRef=match[1];sup.contentEditable='false';
                fragment.append(sup);start=match.index+match[0].length;
            }
            fragment.append(document.createTextNode(node.nodeValue.slice(start)));node.replaceWith(fragment);
        }
        box.querySelectorAll('.ocr-note-reference').forEach(sup=>{
            const entry=refs.get(sup.dataset.noteRef);sup.textContent=String(entry?.number || '?');
            sup.title=entry?.link?.note_id ? '注釈番号（右クリックで関連付けを解除）' : '対応する注釈文を選択してください';
            sup.oncontextmenu=event=>{event.preventDefault();ocrRemoveNoteReference(sup.dataset.noteRef);};
        });
        box.onclick=ocrCaptureNotePosition;
    }
    document.querySelectorAll('.ocr-note-card-number').forEach(badge=>{
        const matches=[...refs.values()].filter(r=>r.link && Number(r.link.note_page)===Number(badge.dataset.notePage) && String(r.link.note_id)===badge.dataset.noteId);
        badge.textContent=matches.map(r=>r.number).join('・');badge.hidden=!matches.length;
    });
}
function ocrPersistNotePage(page) {
    const doc=ocrNoteDoc(page);if(!doc)return;
    markPageModified(page);
    if(page===currentPage)saveCurrentPageToCache(true);
    else {pageCache[page].isModified=true;debounceSavePageToDisk(page,doc,400);}
}
async function ocrToggleNoteInsertion() {
    if(!currentPdfFile)return;
    if(ocrNotePending){ocrRemoveNoteReference(ocrNotePending.id);ocrNotePending=null;}
    ocrNoteArmed=!ocrNoteArmed;
    if(ocrNoteArmed) {
        try {
            await ocrLoadNoteCatalog();
        }catch(error){ocrNoteArmed=false;alert('注釈一覧を読み込めませんでした。アプリを再起動してお試しください。');return;}
        switchTab('body');
    }
    ocrNoteStatus(ocrNoteArmed?'本文の番号を入れる位置をクリックしてください。':'');ocrRefreshNoteNumbers();
}
function ocrNoteStatus(text) {
    document.getElementById('ocrNoteStatus').textContent=text;
    document.getElementById('btnInsertOcrNote').classList.toggle('active',ocrNoteArmed || !!ocrNotePending);
}
function ocrCaptureNotePosition(event) {
    if(!ocrNoteArmed)return;
    const box=document.getElementById('richTextBox1'),selection=window.getSelection();
    if(!selection.rangeCount || !box.contains(selection.anchorNode) || event.target.closest('.page-anchor-header,.ocr-note-reference'))return;
    const range=selection.getRangeAt(0).cloneRange();range.collapse(true);
    const prefix=range.cloneRange();prefix.selectNodeContents(box);prefix.setEnd(range.startContainer,range.startOffset);
    const headers=[...prefix.toString().matchAll(/---\s*ページ\s*(\d+)\s*---/g)];
    const page=headers.length?Number(headers.at(-1)[1]):currentPage,doc=ocrNoteDoc(page);
    if(!doc){alert('対象ページを開いてから挿入してください。');return;}
    const id=crypto.randomUUID(),token=document.createTextNode(`⟦注:${id}⟧`);
    range.insertNode(token);doc.footnote_links ||= [];doc.footnote_links.push({id,note_page:null,note_id:null});
    ocrNotePending={id,page,filename:currentPdfFile.name};ocrNoteArmed=false;
    onBodyTextInput();ocrPersistNotePage(page);ocrRefreshNoteNumbers();
    ocrNoteStatus('対応する注釈文ブロックをクリックしてください。ページを移動して選ぶこともできます。');
    switchTab('footnote');
}
function ocrBindNote(page,index) {
    if(!ocrNotePending || ocrNotePending.filename!==currentPdfFile?.name)return;
    const doc=ocrNoteDoc(page),source=ocrNoteDoc(ocrNotePending.page);if(!doc || !source)return;
    const noteRegions=(doc.regions || []).filter(r=>r.type==='footnote').sort((a,b)=>(a.reading_order || 0)-(b.reading_order || 0));
    doc.footnotes=(doc.footnotes || []).map((note,i)=>{
        const result=typeof note==='string'?{text:note,note_id:`note-${i}`}:{...note,note_id:ocrNoteKey(note,i)};
        if(!result.id && !result.region_id && noteRegions[i])result.region_id=noteRegions[i].id;
        return result;
    });
    const note=doc.footnotes[index],link=(source.footnote_links || []).find(l=>l.id===ocrNotePending.id);if(!note || !link)return;
    link.note_page=page;link.note_id=ocrNoteKey(note,index);
    ocrPersistNotePage(page);ocrPersistNotePage(ocrNotePending.page);
    ocrNotePending=null;ocrNoteStatus('注釈文を関連付けました。');ocrRefreshNoteNumbers();
}
function ocrRemoveNoteReference(id) {
    const refs=ocrNoteReferences(),entry=refs.get(id);if(!entry)return;
    const doc=ocrNoteDoc(entry.page);if(!doc)return;
    doc.body_text=(doc.body_text || '').replaceAll(`⟦注:${id}⟧`,'');
    doc.footnote_links=(doc.footnote_links || []).filter(l=>l.id!==id);
    document.querySelectorAll('.ocr-note-reference').forEach(n=>{if(n.dataset.noteRef===id)n.remove();});
    if(ocrNotePending?.id===id)ocrNotePending=null;
    ocrPersistNotePage(entry.page);ocrRefreshNoteNumbers();ocrNoteStatus('');
}
const ocrOriginalBodyRender=renderBodyText;
renderBodyText=function(...args){ocrOriginalBodyRender(...args);ocrRefreshNoteNumbers();ocrLoadNoteCatalog().catch(()=>{});};
const ocrOriginalBodyInput=onBodyTextInput;
onBodyTextInput=function(...args){ocrOriginalBodyInput(...args);ocrRefreshNoteNumbers();};
const ocrOriginalNotesRender=renderFootnotes;
renderFootnotes=function(notes) {
    ocrDisplayedNotes=notes || [];ocrOriginalNotesRender(notes);
    document.querySelectorAll('#footnoteContainer .footnote-card').forEach((card,index)=>{
        const note=notes[index],page=Number(note?._source_page || currentPage),sourceIndex=note?._source_index ?? index;
        const source=ocrNoteDoc(page)?.footnotes?.[sourceIndex] ?? note,key=ocrNoteKey(source,sourceIndex);
        card.style.position='relative';card.style.paddingRight='45px';
        const badge=document.createElement('span');badge.className='ocr-note-card-number';badge.dataset.notePage=page;badge.dataset.noteId=key;card.append(badge);
        card.addEventListener('click',event=>{if(ocrNotePending){event.preventDefault();event.stopPropagation();ocrBindNote(page,sourceIndex);}},true);
    });ocrRefreshNoteNumbers();
};
onFootnoteInput=function(index,value) {
    const display=ocrDisplayedNotes[index],page=Number(display?._source_page || currentPage),i=display?._source_index ?? index,doc=ocrNoteDoc(page);
    if(!doc?.footnotes?.[i])return;
    const text=value.replace(/^\[P\d+\]\s*/,'');
    if(typeof doc.footnotes[i]==='object')doc.footnotes[i].text=text;else doc.footnotes[i]=text;
    const note=doc.footnotes[i],key=ocrNoteKey(note,i),region=(doc.regions || []).find(r=>r.type==='footnote' && (r.id===key || r.id===note.region_id || r.id===note.id));if(region)region.text=text;
};
saveFootnoteChange=function(index,value){onFootnoteInput(index,value);ocrPersistNotePage(Number(ocrDisplayedNotes[index]?._source_page || currentPage));};
const ocrOriginalDeleteNote=deleteFootnote;
deleteFootnote=function(index){
    const display=ocrDisplayedNotes[index],page=Number(display?._source_page || currentPage),i=display?._source_index ?? index,doc=ocrNoteDoc(page),key=ocrNoteKey(doc?.footnotes?.[i],i);
    if([...ocrNoteReferences().values()].some(r=>r.link && Number(r.link.note_page)===page && r.link.note_id===key)){alert('先に本文の注釈番号を右クリックして関連付けを解除してください。');return;}
    if(page===currentPage && !display?._source_page)return ocrOriginalDeleteNote(index);
    if(doc){doc.footnotes.splice(i,1);ocrPersistNotePage(page);renderFootnotes(doc.footnotes);}
};
document.addEventListener('DOMContentLoaded',()=>{
    const bar=document.getElementById('pnlBodyTextToolbar') || document.getElementById('tabBar');
    const button=document.createElement('button');button.id='btnInsertOcrNote';button.className='table-btn';button.textContent='注¹ 注釈番号';button.title='本文の位置、注釈文の順にクリック（再クリックで中止）';button.onclick=ocrToggleNoteInsertion;bar.append(button);
    const status=document.createElement('div');status.id='ocrNoteStatus';status.style.cssText='color:#166534; font-size:12px; padding:4px 12px;';document.getElementById('tabOcrResultContainer').prepend(status);
    ocrRefreshNoteNumbers();ocrLoadNoteCatalog().catch(()=>{});
});
