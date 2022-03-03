import cv2
import numpy as np
import sys, os

SIZE=96

def resize(img):
    h, w, c = img.shape
    if w > h:
        nw, nh = SIZE, int(h * SIZE/w)
        if nh < 10 : nh = 10
        img = cv2.resize(img, (nw, nh))
        a1 = int((SIZE-nh)/2)
        a2= SIZE-nh-a1
        pad1 = np.zeros((a1, SIZE, c), dtype=np.uint8)
        pad2 = np.zeros((a2, SIZE, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=0)
    else:
        nw, nh = int(w * SIZE/h), SIZE
        if nw < 10 : nw = 10
        img = cv2.resize(img, (nw, nh))
        a1 = int((SIZE-nw)/2)
        a2= SIZE-nw-a1
        pad1 = np.zeros((SIZE, a1, c), dtype=np.uint8)
        pad2 = np.zeros((SIZE, a2, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=1)
    return img


def adding_guass(image, param=30, grayscale=255):
    w=image.shape[1]
    h=image.shape[0]
    for i in range(3):
        img = image[:,:,i]
        newimg=np.zeros((h,w),np.uint8)
        for x in range(0,h):
            for y in range(0,w-1,2):
                r1=np.random.random_sample()
                r2=np.random.random_sample()
                z1=param*np.cos(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
                z2=param*np.sin(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))

                fxy=int(img[x,y]+z1)
                fxy1=int(img[x,y+1]+z2)
                #f(x,y)
                if fxy<0:
                    fxy_val=0
                elif fxy>grayscale:
                    fxy_val=grayscale
                else:
                    fxy_val=fxy
                #f(x,y+1)
                if fxy1<0:
                    fxy1_val=0
                elif fxy1>grayscale:
                    fxy1_val=grayscale
                else:
                    fxy1_val=fxy1
                newimg[x,y]=fxy_val
                newimg[x,y+1]=fxy1_val

        image[:,:,i] = newimg
    return image

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rot_mat = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)

    return rot_mat, cv2.warpAffine(src, rot_mat, (w, h), flags=cv2.INTER_LANCZOS4)


def doAugmentation(imgpath, label):
    pre, suf = imgpath.split('.')
    raw_img = cv2.imread(imgpath)
    raw_img = resize(raw_img)
    #for i in [1,2,3,4, 87,88,89,90,91,92,93, 177,178,179,180,181,182,183, 267,268,269,270,271,272,273, 357,358,359]:
    #for i in [1,2,3,4,5,6,7,8,9,10,350,351,352,353,354,355,356,357,358,359]:
    for i in [0,1,2,3,4,5,355,356,357,358,359]:
        _, rot_img = rotate_about_center(raw_img, i)
        rot_imgpath = pre+'_rot%03d.'%i+suf
        print(rot_imgpath)
        f.write(rot_imgpath+"\t"+label+"\n")
        cv2.imwrite(rot_imgpath, rot_img)

        img = cv2.GaussianBlur(rot_img, (3,3), 2)
        imp = pre+'_rot%03d'%i+'_blur3.'+suf
        print(imp)
        f.write(imp+"\t"+label+"\n")
        cv2.imwrite(imp, img)
        
        img = cv2.GaussianBlur(rot_img, (5,5), 2)
        imp = pre+'_rot%03d'%i+'_blur5.'+suf
        print(imp)
        f.write(imp+"\t"+label+"\n")
        cv2.imwrite(imp, img)

        img = adding_guass(rot_img)
        imp = pre+'_rot%03d'%i+'_guas1.'+suf
        print(imp)
        f.write(imp+"\t"+label+"\n")
        cv2.imwrite(imp, img)

        img = adding_guass(rot_img)
        img = adding_guass(img)
        imp = pre+'_rot%03d'%i+'_guas2.'+suf
        print(imp)
        f.write(imp+"\t"+label+"\n")
        cv2.imwrite(imp, img)

        img = adding_guass(cv2.GaussianBlur(rot_img, (3,3), 2))
        imp = pre+'_rot%03d'%i+'_guasblur3.'+suf
        print(imp)
        f.write(imp+"\t"+label+"\n")
        cv2.imwrite(imp, img)

        img = cv2.GaussianBlur(adding_guass(rot_img), (3,3), 2)
        imp = pre+'_rot%03d'%i+'_blur3guas.'+suf
        print(imp)
        f.write(imp+"\t"+label+"\n")
        cv2.imwrite(imp, img)


def main():
    for line in open(flist).readlines():
        imp, label = line.strip().split('\t')
        print(imp, label)
        doAugmentation(imp, label)



if __name__=='__main__':
    flist = sys.argv[1]
    with open(flist+'.aug', 'w') as f:
        main()
