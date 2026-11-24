import cv2, numpy as np, sys
S=sys.argv[1]
W={1:212,2:150,3:155,4:195,5:148,6:180,7:130,8:186}
H={1:94,2:96,3:94,4:96,5:96,6:96,7:96,8:96}
rng=np.random.default_rng(1)
for i in range(1,9):
    im=cv2.imread(f'{S}/panels/p{i}.png')
    h,w=im.shape[:2]
    m=np.zeros((h,w),np.uint8); m[0:H[i],0:W[i]]=255
    # multi-scale inpaint: fill at low res for smooth bokeh, then blend
    small=cv2.resize(im,(w//4,h//4),interpolation=cv2.INTER_AREA)
    ms=cv2.resize(m,(w//4,h//4),interpolation=cv2.INTER_NEAREST)
    fs=cv2.inpaint(small,cv2.dilate(ms,np.ones((3,3))),9,cv2.INPAINT_TELEA)
    fs=cv2.GaussianBlur(fs,(0,0),2)
    up=cv2.resize(fs,(w,h),interpolation=cv2.INTER_CUBIC).astype(np.float32)
    up+=rng.normal(0,2.2,up.shape)  # match grain
    a=cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),6)
    a=np.maximum(a,(m>0)*1.0)
    a=cv2.GaussianBlur(a,(0,0),4)[...,None]
    out=im*(1-a)+up*a
    cv2.imwrite(f'{S}/clean/p{i}.png',np.clip(out,0,255).astype(np.uint8))
