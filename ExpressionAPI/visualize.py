import numpy as np
from skimage import draw
from skimage import color 
from skimage.transform import rescale

def _add_mask(img, pts):
    if img.shape[-1]==1:
        img = gray2rgb(img[:,:,0])
    radius = 3
    for x,y in pts:
        x = np.round(x)
        y = np.round(y)
        rr, cc = draw.circle(y, x, radius)
        try:
            img[rr, cc, 0] = 0
            img[rr, cc, 1] = 1
            img[rr, cc, 2] = 0
        except IndexError:
            pass
    return img

def make_hub(img, pts, au_disfa, img_bb=None, h=640, margin=224):

    scale = h/img.shape[0]
    img = rescale(img, scale)

    img_pad = []
    for i in range(3):
        tmp = np.pad(img[:,:,i],margin,mode='constant',constant_values=0)
        img_pad.append(tmp)
    img = np.stack(img_pad,axis=-1)
    pts = (pts*scale) + margin

    img = _add_mask(img, pts)

    AUs = {
            'eye_brow':[max(au_disfa[:2]),pts[24],(0,-1)],
        # '01':[au_disfa[0],pts[23],(0,-1)],
        # '02':[au_disfa[1],pts[25],(0,-1)],
        '04':[au_disfa[2],(pts[21]+pts[22])/2,(-1,-1)],
        '06':[au_disfa[4],(pts[16]+pts[45])/2,(1,0)],
        '17':[au_disfa[8],(pts[58]+pts[9])/2,(1,1)],
        '25':[au_disfa[10],(pts[51]+pts[57])/2,(-1,0)],
        '12':[au_disfa[6],pts[54],(1,0)],
        }

    for n in AUs:
        bar = 75
        I, [p0,p1], [t0,t1] =  AUs[n]
        if I<0:I=0
        p0 = int(np.round(p0))
        p1 = int(np.round(p1))
        rr, cc = draw.circle(p1, p0, 6)
        img[rr, cc, 0] = 1
        img[rr, cc, 1] = 0
        img[rr, cc, 2] = 0
        bar = int(bar/(t0**2+t1**2)**0.5)
        rr, cc = draw.line(p1, p0, p1+t1*bar, p0+t0*bar)
        for i in range(-3,3):
            for j in range(-3,3):
                for c in range(3):
                    img[rr+i, cc+j, c]=0

        rr, cc = draw.line(p1, p0, int(p1+t1*bar/5*I), int(p0+t0*bar/5*I))
        for i in range(-5,5):
            for j in range(-5,5):
                for c in range(3):
                    img[rr+i, cc+j, 0]=1

    if img_bb!=None:
        img_bb-=img_bb.min()
        img_bb/=img_bb.max()
        img_bb = rescale(img_bb, margin/img_bb.shape[0])
        img[margin:2*margin:,-margin:]=img_bb
        print(img_bb.shape)

    return img
