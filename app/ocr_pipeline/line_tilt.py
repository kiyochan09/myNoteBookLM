"""Conservative local deskew for scanned vertical prose without changing page size."""
import base64
import cv2
import numpy as np


def decode_page_image(value, shape=None):
    try:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(value.split(',', 1)[-1], validate=True), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, TypeError) as exc:
        raise ValueError('画像を読み取れません。') from exc
    if image is None or (shape is not None and image.shape[:2] != tuple(shape)):
        raise ValueError('補正画像の大きさが元のページと一致しません。')
    return image


def correct_vertical_tilt(image, boundaries=None):
    height, width = image.shape[:2]
    boundaries=boundaries or []
    if len(boundaries)>40:
        raise ValueError('分割線は40本以内にしてください。')
    try:
        boundaries=sorted([(float(b['top']),float(b['bottom'])) for b in boundaries],key=lambda b:b[0])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('分割線の位置を確認してください。') from exc
    if any(not np.isfinite(a+b) or not 0<a<width or not 0<b<width for a,b in boundaries) or any(
        boundaries[i][0]>=boundaries[i+1][0] or boundaries[i][1]>=boundaries[i+1][1] for i in range(len(boundaries)-1)):
        raise ValueError('分割線が交差しています。位置を調整してください。')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    max_size = min(height, width) * .07
    glyphs = np.array([(x+w/2, y+h/2, w, h) for x,y,w,h,a in stats[1:]
                       if a >= 12 and 7 <= w <= max_size and 7 <= h <= max_size], dtype=np.float32)
    if len(glyphs) < 30:
        raise ValueError('傾きを測れる縦書きの文字が見つかりません。')
    gw, gh = np.median(glyphs[:, 2:], axis=0)
    glyphs = glyphs[(glyphs[:,2] >= gw*.65) & (glyphs[:,2] <= gw*1.8)
                    & (glyphs[:,3] >= gh*.65) & (glyphs[:,3] <= gh*1.8)]
    # Neighboring character centers establish a starting lean for column grouping.
    def band_at(x,y):
        return int(np.searchsorted([a+(b-a)*y/height for a,b in boundaries],x))
    bands=np.array([band_at(g[0],g[1]) for g in glyphs])
    slopes={i:[] for i in range(len(boundaries)+1)}
    for index,g in enumerate(glyphs):
        dx=glyphs[:,0]-g[0];dy=glyphs[:,1]-g[1]
        possible=np.flatnonzero((dy >= gh*.65) & (dy <= gh*1.8) & (np.abs(dx) <= gw*.65) & (bands==bands[index]))
        if len(possible):
            j=possible[np.argmin(np.abs(dx[possible]))];slopes[bands[index]].append(float(dx[j]/dy[j]))
    leans={i:float(np.clip(np.median(values),-.15,.15)) if values else 0. for i,values in slopes.items()}
    projected=glyphs[:,0]-np.array([leans[i] for i in bands])*(glyphs[:,1]-height/2)
    groups=[]
    for i in sorted(range(len(glyphs)),key=lambda i:(bands[i],projected[i])):
        if not groups or bands[i]!=bands[groups[-1][0]] or projected[i]-np.median(projected[groups[-1]]) > gw*.85:
            groups.append([i])
        else:
            groups[-1].append(i)
    columns=[]
    for indices in groups:
        points=glyphs[indices,:2]
        if len(points) < 8 or np.ptp(points[:,1]) < height*.35:
            continue
        vx,vy,x,y=cv2.fitLine(points,cv2.DIST_HUBER,0,.01,.01).flatten()
        if abs(float(vy)) < .5:
            continue
        slope=float(vx/vy)
        if abs(slope) > .15:
            continue
        center=float(x+slope*(height/2-y))
        columns.append((center,slope,float(points[:,1].min()-gh),float(points[:,1].max()+gh),int(bands[indices[0]])))
    columns.sort()
    if not columns:
        raise ValueError('傾きを測れる縦書きの行が見つかりません。')
    selected=columns
    centers=np.array([c[0] for c in selected])
    # User lines define the independent correction bands.
    segment_ids=[c[4] for c in selected]
    local=np.array([np.median([other[1] for other,band in zip(selected,segment_ids) if band==segment]) for segment in segment_ids])
    # Resample the original image once. Never erase or move character components:
    # one character may comprise multiple disconnected strokes or touch a neighbor.
    xx=np.arange(width,dtype=np.float32)
    field=np.interp(xx,centers,local).astype(np.float32)
    # Keep the coordinate map monotone. A steep change at a split would fold
    # the image and duplicate, cut, or crush neighboring characters.
    sigma=2.0
    while float(np.max(np.abs(np.diff(field)))) * height/2 > .25:
        field=cv2.GaussianBlur(field[None,:],(0,0),sigmaX=sigma,sigmaY=1,borderType=cv2.BORDER_REPLICATE)[0]
        sigma*=2
        if sigma>width*2:
            raise ValueError("この分割では変形が大きすぎます。分割線を文字のない行間へ移してください。")
    # The median column start can lie inside earlier first characters. Use
    # the earliest measured start, then place transitions entirely in blank
    # rows across the text area. Never ramp displacement through a glyph.
    y0=max(0,int(np.floor(min(c[2] for c in selected))))
    y1=min(height,int(np.ceil(max(c[3] for c in selected)))+1)
    fade=max(8,int(gh*.65))
    text_x0=max(0,int(np.floor(centers.min()-gw)))
    text_x1=min(width,int(np.ceil(centers.max()+gw))+1)
    blank=~np.any(binary[:,text_x0:text_x1]>0,axis=1)
    edges=np.diff(np.r_[False,blank,False].astype(np.int8))
    runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    top_runs=[(a,min(b,y0)) for a,b in runs if min(b,y0)-a>=fade+4]
    bottom_runs=[(max(a,y1),b) for a,b in runs if b-max(a,y1)>=fade+4]
    # Without a safe gap, continue the gentle shear to the page edge instead
    # of creating a sharp local stretch in printed strokes.
    top_end=int(top_runs[-1][1])-2 if top_runs else 0
    bottom_start=int(bottom_runs[0][0])+2 if bottom_runs else height
    yy=np.arange(height,dtype=np.float32)
    weight=np.minimum(np.clip((yy-(top_end-fade))/fade,0,1),np.clip(((bottom_start+fade)-yy)/fade,0,1))
    map_y=np.broadcast_to(yy[:,None],(height,width)).copy()
    shift=field[None,:]*(map_y-height/2)*weight[:,None]
    map_x=(np.broadcast_to(xx,(height,width)).copy()+shift).astype(np.float32)
    derivative=np.diff(map_x,axis=1)
    if float(derivative.min())<.7 or float(derivative.max())>1.3:
        raise ValueError("文字を保てる補正範囲を超えました。分割線を調整してください。")
    result=cv2.remap(image,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    untouched=weight==0
    result[untouched]=image[untouched]
    return result, {'detected_lines':len(columns),'selected_lines':len(selected),
                    'segments':len(boundaries)+1,'measured_segments':len(set(segment_ids)),
                    'range_x':[0,width], 'range_y':[max(0,top_end-fade),min(height,bottom_start+fade)],
                    'min_horizontal_scale':round(float(derivative.min()),3),
                    'max_horizontal_scale':round(float(derivative.max()),3),
                    'max_shift':round(float(np.max(np.abs(shift))),1)}



def correct_spread_tilt(image):
    """Measure and resample each scanned page independently at the fold."""
    height, width = image.shape[:2]
    if width < height * 1.05:
        raise ValueError('横長の見開き画像で実行してください。単ページには「傾きを補正」を使用します。')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = np.maximum(0.0, 240.0 - gray).sum(axis=0)
    kernel = max(5, min(25, (width // 40) | 1))
    smooth = cv2.GaussianBlur(profile[None, :], (kernel, 1), 5.0).ravel()
    start, end = int(width * .44), int(width * .56)
    split = start + int(np.argmin(smooth[start:end]))
    if abs(split - width // 2) > width * .025:
        split = width // 2
    result = image.copy()
    pages = []
    for side, x0, x1 in [('left', 0, split), ('right', split, width)]:
        try:
            corrected, info = correct_vertical_tilt(image[:, x0:x1])
            result[:, x0:x1] = corrected
            pages.append(dict(info, side=side, corrected=True))
        except ValueError as exc:
            # A blank page or illustration must not prevent the prose page
            # from being corrected, nor be distorted by its neighbor's slope.
            pages.append(dict(side=side, corrected=False, reason=str(exc)))
    measured = sum(page['corrected'] for page in pages)
    if not measured:
        raise ValueError('左右のページに傾きを測れる縦書きの行が見つかりません。')
    return result, dict(mode='spread', segments=2, measured_segments=measured,
                        gutter_x=split, pages=pages,
                        detected_lines=sum(page.get('detected_lines', 0) for page in pages))


def correct_image_request(value, boundaries=None, mode='manual'):
    image = decode_page_image(value)
    if mode == 'spread':
        result, info = correct_spread_tilt(image)
    elif mode == 'manual':
        result, info = correct_vertical_tilt(image, boundaries)
    else:
        raise ValueError('傾き補正の方式を確認してください。')
    ok, png = cv2.imencode('.png', result)
    if not ok:
        raise ValueError('補正画像を作成できません。')
    return dict(info, image_base64='data:image/png;base64,' + base64.b64encode(png).decode('ascii'))
