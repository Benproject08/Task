import cv2, numpy as np, subprocess, sys
S, OUT = sys.argv[1], sys.argv[2]
FPS=24; OW,OH=1920,1080; AW,AH=1920,804; AR=AW/AH; UP=4
ease=lambda t: t*t*(3-2*t)*0.6+t*0.4   # gentle ease, never fully stops
# (zoom_start, zoom_end, (cx,cy) start, (cx,cy) end) in normalized panel coords
shots={
1:(1.00,1.16,(0.50,0.55),(0.62,0.62)),   # establishing: push in toward her
2:(1.08,1.14,(0.44,0.52),(0.54,0.50)),   # medium: lateral dolly
3:(1.04,1.14,(0.50,0.50),(0.52,0.46)),   # close-up: slow push on face
4:(1.12,1.08,(0.62,0.48),(0.56,0.48)),   # over-shoulder: arc/pan
5:(1.08,1.14,(0.54,0.52),(0.47,0.48)),   # profile: tracking
6:(1.06,1.16,(0.52,0.50),(0.56,0.60)),   # detail: push to hands/watch
7:(1.10,1.06,(0.50,0.62),(0.52,0.40)),   # low angle: tilt up
8:(1.16,1.04,(0.56,0.45),(0.55,0.48)),   # final: slow pull back
}
yy,xx=np.mgrid[0:AH,0:AW].astype(np.float32)
vig=1-0.28*(((xx-AW/2)/(AW/2))**2*0.6+((yy-AH/2)/(AH/2))**2*0.4)
vig=np.clip(vig,0,1)[...,None]
rng=np.random.default_rng(7)
ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{OW}x{OH}','-r',str(FPS),'-i','-',
  '-c:v','libx264','-preset','slow','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',OUT],stdin=subprocess.PIPE)
total=8*FPS; n=0
for i in range(1,9):
    src=cv2.imread(f'{S}/clean/p{i}.png'); h,w=src.shape[:2]
    big=cv2.resize(src,(w*UP,h*UP),interpolation=cv2.INTER_LANCZOS4)
    bl=cv2.GaussianBlur(big,(0,0),2.0); big=cv2.addWeighted(big,1.35,bl,-0.35,0)  # mild detail restore
    H_,W_=h*UP,w*UP; base_w=min(W_,H_*AR)
    z0,z1,c0,c1=shots[i]
    for f in range(FPS):
        t=ease(f/(FPS-1)); z=z0+(z1-z0)*t
        vw=base_w/z; vh=vw/AR
        cx=(c0[0]+(c1[0]-c0[0])*t)*W_; cy=(c0[1]+(c1[1]-c0[1])*t)*H_
        # subtle handheld drift
        cx+=np.sin(n*0.21+i)*vw*0.002; cy+=np.cos(n*0.17+i*2)*vh*0.003
        cx=np.clip(cx,vw/2,W_-vw/2); cy=np.clip(cy,vh/2,H_-vh/2)
        s=AW/vw
        M=np.float32([[s,0,-(cx-vw/2)*s],[0,s,-(cy-vh/2)*s]])
        fr=cv2.warpAffine(big,M,(AW,AH),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        fr=fr*vig+rng.normal(0,3.0,(AH,AW,1)).astype(np.float32)
        # fade in / out
        g=min(1,(n+1)/6, (total-n)/8); fr*=g
        canvas=np.zeros((OH,OW,3),np.uint8); y0=(OH-AH)//2
        canvas[y0:y0+AH]=np.clip(fr,0,255).astype(np.uint8)
        ff.stdin.write(canvas.tobytes()); n+=1
ff.stdin.close(); ff.wait(); print('frames',n)
