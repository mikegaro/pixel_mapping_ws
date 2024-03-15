import cv2
import numpy as np

def adjust_contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    return enhanced_img

def adjust_vibrance(vibrance: float, img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    print(np.amax(s), np.amin(s), s.dtype)

    # create 256 element non-linear LUT for sigmoidal function
    # see https://en.wikipedia.org/wiki/Sigmoid_function
    xval = np.arange(0, 256)
    lut = (255*np.tanh(vibrance*xval/255)/np.tanh(1)+0.5).astype(np.uint8)

    # apply lut to saturation channel
    new_s = cv2.LUT(s,lut)

    # combine new_s with original h and v channels
    new_hsv = cv2.merge([h,new_s,v])

    # convert back to BGR
    result =  cv2.cvtColor(new_hsv,  cv2.COLOR_HSV2RGB)
    return result

def increase_sharpening(cv2_img):
    # create a sharpening kernel
    sharpen_filter=np.array([[-1,-1,-1],
                             [-1,9,-1],
                             [-1,-1,-1]])
    
    # applying kernels to the input image to get the sharpened image
    return cv2.filter2D(cv2_img,-1,sharpen_filter)
