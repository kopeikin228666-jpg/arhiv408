import sys, cv2, numpy as np
out=sys.argv[1]; fs=sys.argv[2:]; ims=[]
for f in fs:
    im=cv2.imread(f); im=cv2.resize(im,(540,960),interpolation=cv2.INTER_AREA)
    s=0.5
    cv2.line(im,(0,int(1540*s)),(540,int(1540*s)),(255,0,255),1)
    cv2.line(im,(0,100),(540,100),(255,0,255),1)
    cv2.rectangle(im,(450,450),(539,850),(255,0,255),1)
    cv2.putText(im,f.split('/')[-1][:-4],(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    ims.append(im)
cv2.imwrite(out,np.hstack(ims),[cv2.IMWRITE_JPEG_QUALITY,90])
