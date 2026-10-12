import re
import colorsys
import xml.etree.ElementTree as ET
A='{http://schemas.openxmlformats.org/drawingml/2006/main}'
WPS='{http://schemas.microsoft.com/office/word/2010/wordprocessingShape}'
V='{urn:schemas-microsoft-com:vml}'

def load_theme(archive):
    if 'word/theme/theme1.xml' not in archive.namelist(): return {}
    root=ET.fromstring(archive.read('word/theme/theme1.xml'))
    scheme=root.find('.//'+A+'clrScheme')
    return {e.tag.split('}')[-1]:next(iter(e)).get('lastClr',next(iter(e)).get('val')) for e in scheme} if scheme is not None else {}

def color(node,theme):
    if node is None:return None
    tag=node.tag.split('}')[-1]; value=node.get('val','')
    if tag=='schemeClr':value=theme.get({'tx1':'dk1','tx2':'dk2','bg1':'lt1','bg2':'lt2'}.get(value,value),'')
    elif tag=='sysClr':value=node.get('lastClr','')
    if not re.fullmatch('[0-9a-fA-F]{6}',value):return None
    rgb=[int(value[i:i+2],16) for i in (0,2,4)]
    for transform in node:
        kind=transform.tag.split('}')[-1]; factor=int(transform.get('val','100000'))/100000
        if kind=='tint':rgb=[round(c+(255-c)*factor) for c in rgb]
        elif kind=='shade':rgb=[round(c*factor) for c in rgb]
        elif kind in ('satMod','lumMod','lumOff'):
            h,l,s=colorsys.rgb_to_hls(*(c/255 for c in rgb))
            if kind=='satMod':s=min(1,max(0,s*factor))
            elif kind=='lumMod':l=min(1,max(0,l*factor))
            else:l=min(1,max(0,l+factor))
            rgb=[round(c*255) for c in colorsys.hls_to_rgb(h,l,s)]
    return '#'+''.join(f'{min(255,max(0,c)):02X}' for c in rgb)

def textbox_style(box,root,theme):
    owner=box
    if hasattr(box,'getparent'):
        while owner.getparent() is not None and owner.tag not in (WPS+'wsp',V+'shape',V+'rect',V+'roundrect'):owner=owner.getparent()
    else:
        parents={child:parent for parent in root.iter() for child in parent}
        while owner in parents and owner.tag not in (WPS+'wsp',V+'shape',V+'rect',V+'roundrect'):owner=parents[owner]
    result=[]
    if owner.tag==WPS+'wsp':
        props=owner.find(WPS+'spPr'); style=owner.find(WPS+'style')
        fill=props.find(A+'solidFill') if props is not None else None
        gradient=props.find(A+'gradFill') if props is not None else None
        nofill=props is not None and props.find(A+'noFill') is not None
        if fill is None and gradient is None and not nofill and style is not None:
            fill=style.find(A+'fillRef')
            if fill is not None and fill.get('idx')=='0':nofill=True
        if nofill:result.append('background-color:transparent')
        elif gradient is not None:
            stops=[]
            for stop in gradient.findall(A+'gsLst/'+A+'gs'):
                resolved=color(stop[0],theme) if len(stop) else None
                if resolved:stops.append(resolved+' '+str(int(stop.get('pos','0'))/1000)+'%')
            angle=gradient.find(A+'lin')
            degrees=(int(angle.get('ang','0'))/60000+90)%360 if angle is not None else 180
            if stops:result.append('background:linear-gradient('+str(degrees)+'deg,'+','.join(stops)+')')
        elif fill is not None and len(fill):
            resolved=color(fill[0],theme)
            if resolved:result.append('background-color:'+resolved)
        font=style.find(A+'fontRef') if style is not None else None
        if font is not None and len(font):
            resolved=color(font[0],theme)
            if resolved:result.append('color:'+resolved)
    elif owner.tag in (V+'shape',V+'rect',V+'roundrect'):
        value=owner.get('fillcolor','')
        if owner.get('filled') in ('f','false'):result.append('background-color:transparent')
        elif re.fullmatch(r'#[0-9a-fA-F]{3,6}|[a-zA-Z]+',value):result.append('background-color:'+value)
    return ';'.join(result)+';' if result else ''
