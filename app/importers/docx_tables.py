"""Read physical Word cells once, preserving horizontal and vertical merges."""
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

def parse_docx_table(element, parse_paragraph):
    physical_rows=[]; active={}; width=len(element.findall(W+'tblGrid/'+W+'gridCol'))
    for ri,tr in enumerate(element.findall(W+'tr')):
        before=tr.find(W+'trPr/'+W+'gridBefore')
        col=int(before.get(W+'val','0')) if before is not None else 0
        cells=[]; next_active={}
        for tc in tr.findall(W+'tc'):
            span_node=tc.find(W+'tcPr/'+W+'gridSpan')
            span=int(span_node.get(W+'val','1')) if span_node is not None else 1
            vm=tc.find(W+'tcPr/'+W+'vMerge')
            continuation=vm is not None and vm.get(W+'val','continue')!='restart'
            origin=active.get(col)
            if continuation and origin is not None and origin['colspan']==span:
                origin['rowspan']+=1
                for ci in range(col,col+span):next_active[ci]=origin
            else:
                plain=[]; rich=[]
                for p in tc.findall(W+'p'):
                    text,markup,*_=parse_paragraph(p)
                    if text:plain.append(text)
                    if markup:rich.append(markup)
                cell={'row':ri,'column':col,'colspan':span,'rowspan':1,'text':' '.join(plain).strip(),'html':'<br>'.join(rich)}
                cells.append(cell)
                if vm is not None:
                    for ci in range(col,col+span):next_active[ci]=cell
            col+=span
        active=next_active;width=max(width,col);physical_rows.append(cells)
    rows=[['']*width for _ in physical_rows]
    for cells in physical_rows:
        for cell in cells:rows[cell['row']][cell['column']]=cell['text']
    grid={'headers':rows[0] if rows else [],'rows':rows,'columns':width,'row_count':len(rows),
          'cell_spans':[{k:c[k] for k in ('row','column','rowspan','colspan')} for cells in physical_rows for c in cells]}
    parts=['<div class="table-embed"><table class="styled-table"><tbody>']
    covered=set()
    for ri,cells in enumerate(physical_rows):
        parts.append('<tr>');col=0
        for cell in cells:
            while col<cell['column']:
                if (ri,col) not in covered:parts.append('<td></td>')
                col+=1
            tag='th' if ri==0 else 'td'
            attrs=''.join(f' {name}="{cell[name]}"' for name in ('colspan','rowspan') if cell[name]>1)
            parts.append(f'<{tag}{attrs}>{cell["html"]}</{tag}>')
            for rr in range(ri+1,ri+cell['rowspan']):
                for cc in range(cell['column'],cell['column']+cell['colspan']):covered.add((rr,cc))
            col=cell['column']+cell['colspan']
        while col<width:
            if (ri,col) not in covered:parts.append('<td></td>')
            col+=1
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return grid,''.join(parts)
