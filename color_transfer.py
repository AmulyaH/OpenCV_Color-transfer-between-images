'''
Name : Amulya Huchachar
PSU ID : 
'''

import cv2
import numpy as np
import sys

MRGB_to_LMS = np.array([[0.3811, 0.5783, 0.0402],
                              [0.1967, 0.7244, 0.0782],
                              [0.0241, 0.1288, 0.8444]])

x = np.array([[(1/np.sqrt(3)), 0, 0], 
            [0,(1/np.sqrt(6)),0], 
            [0, 0, (1/np.sqrt(2))]]) 

y = np.array([[1,1, 1],[1,1,-2],[1, -1,0]])

Matrix_LMS_To_LAB = np.matmul(x , y)

TMatrix_LMS_To_RGB = np.array([[4.4679, -3.5873, 0.1193],
                              [-1.2186, 2.3809, -0.1624],
                              [0.0497, -0.2439, 1.2045]])

a = np.array([[1,1, 1],[1,1,-1],[1, -2,0]])

b = np.array([[(np.sqrt(3)/3), 0, 0], 
                [0,(np.sqrt(6)/6),0],  
                [0, 0, (np.sqrt(2)/2)]]) 

Matrix_LAB_To_LMS = np.matmul( a ,  b)
                              
Matrix_LMS_To_CIECAM97s = np.array([[2.00, 1.00, 0.05],
                            [1.00, -1.09, 0.09],
                            [0.11, 0.11, -0.22]])

TMatrix_CIECAM97s_to_LMS = np.linalg.inv(Matrix_LMS_To_CIECAM97s)

def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR,dtype=np.float32)
    
    img_RGB[:,:,0] = img_BGR[:,:,2]
    img_RGB[:,:,1] = img_BGR[:,:,1]
    img_RGB[:,:,2] = img_BGR[:,:,0] 

    return img_RGB

def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB,dtype=np.float32)
    
    img_BGR[:,:,0] = img_RGB[:,:,2]
    img_BGR[:,:,1] = img_RGB[:,:,1]
    img_BGR[:,:,2] = img_RGB[:,:,0]

    return img_BGR

def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    img_LMS = np.zeros_like(img_RGB,dtype=np.float32)
    img_Lab = np.zeros_like(img_RGB,dtype=np.float32)
    x,y,z = img_RGB.shape

    for r in range(x):
        for c in range(y):
            p = img_RGB[r,c,:]
            p = np.array(p)
            img_LMS[r,c,:] = np.dot(MRGB_to_LMS , p) 
            img_LMS[r,c,:] = np.log10(img_LMS[r,c,:])
            img_Lab[r,c,:] = np.dot(Matrix_LMS_To_LAB , img_LMS[r,c,:]) 

    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    img_LMS = np.zeros_like(img_Lab,dtype=np.float32)
    img_RGB = np.zeros_like(img_Lab,dtype=np.float32)
    x,y,z = img_Lab.shape
   
    for r in range(x):
        for c in range(y):
            p = img_Lab[r,c,:]
            p = np.array(p)
            img_LMS[r,c,:] = np.dot(Matrix_LAB_To_LMS , p)
            img_LMS[r,c,:] = np.power(10, img_LMS[r,c,:] )
            img_RGB[r,c,:] = np.dot(TMatrix_LMS_To_RGB , img_LMS[r,c,:] )   
    return img_RGB

def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    img_CIECAM97s = np.zeros_like(img_RGB,dtype=np.float32)  
    x,y,z = img_RGB.shape

    for r in range(x):
        for c in range(y):
            p = img_RGB[r,c,:]
            p = np.array(p)
            p = np.dot(MRGB_to_LMS , p) 
            img_CIECAM97s[r,c,:] = np.dot(Matrix_LMS_To_CIECAM97s, p) 
    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_RGB = np.zeros_like(img_CIECAM97s,dtype=np.float32)
    x,y,z = img_CIECAM97s.shape

    for r in range(x):
        for c in range(y):
            p = img_CIECAM97s[r,c,:]
            p = np.array(p)
            p = np.dot(TMatrix_CIECAM97s_to_LMS , p)
            img_RGB[r,c,:] = np.dot(TMatrix_LMS_To_RGB , p) 
    return img_RGB

def image_stats(img):
	# compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(img)

    (lMean, lStd) = (np.mean(l), np.std(l))
    (aMean, aStd) = (np.mean(a), np.std(a))
    (bMean, bStd) = (np.mean(b), np.std(b))
	# return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    
    img_Lab_Source = convert_color_space_RGB_to_Lab(img_RGB_source)
    img_Lab_Target = convert_color_space_RGB_to_Lab(img_RGB_target)

    # compute color statistics for the source and target images
    (lMeanSource, lStdSource, aMeanSource, aStdSource, bMeanSource, bStdSource) = image_stats(img_Lab_Source)
    (lMeanTarget, lStdTarget, aMeanTarget, aStdTarget, bMeanTarget, bStdTarget) = image_stats(img_Lab_Target)

    (l, a, b) = cv2.split(img_Lab_Source)
    l -= lMeanSource
    a -= aMeanSource
    b -= bMeanSource
  
    # scale by the standard deviations
    l = (lStdTarget / lStdSource) * l
    a = (aStdTarget / aStdSource) * a
    b = (bStdTarget / bStdSource) * b

    # add in the source mean
    l += lMeanTarget
    a += aMeanTarget
    b += bMeanTarget

    # merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer datatype
    result =  cv2.merge([l, a, b])
    img_RGB = convert_color_space_Lab_to_RGB(result)
    (r,g,b) = cv2.split(img_RGB)

    r = np.clip(r, 0, 255.0)
    g = np.clip(g, 0, 255.0)
    b = np.clip(b, 0, 255.0)

    img_RGB =  cv2.merge([r, g, b])
    img_RGB = np.uint8(img_RGB)

    return img_RGB

def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')

     # compute color statistics for the source and target images
    (lMeanSource, lStdSource, aMeanSource, aStdSource, bMeanSource, bStdSource) = image_stats(img_RGB_source)
    (lMeanTarget, lStdTarget, aMeanTarget, aStdTarget, bMeanTarget, bStdTarget) = image_stats(img_RGB_target)

    (r, g, b) = cv2.split(img_RGB_source)
    r -= lMeanSource
    g -= aMeanSource
    b -= bMeanSource
  
    # scale by the standard deviations
    r = (lStdTarget / lStdSource) * r
    g = (aStdTarget / aStdSource) * g
    b = (bStdTarget / bStdSource) * b

    # add in the source mean
    r += lMeanTarget
    g += aMeanTarget
    b += bMeanTarget
    
    r = np.clip(r, 0, 255.0)
    g = np.clip(g, 0, 255.0)
    b = np.clip(b, 0, 255.0) 

    img_RGB =  cv2.merge([r, g, b])
    img_RGB = np.uint8(img_RGB)

    return img_RGB

def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    img_CIECAM97s_Source = convert_color_space_RGB_to_CIECAM97s(img_RGB_source)
    img_CIECAM97s_Target = convert_color_space_RGB_to_CIECAM97s(img_RGB_target)
    
    # compute color statistics for the source and target images
    (lMeanSource, lStdSource, aMeanSource, aStdSource, bMeanSource, bStdSource) = image_stats(img_CIECAM97s_Source)
    (lMeanTarget, lStdTarget, aMeanTarget, aStdTarget, bMeanTarget, bStdTarget) = image_stats(img_CIECAM97s_Target)

    (l, a, b) = cv2.split(img_CIECAM97s_Source)
    l -= lMeanSource
    a -= aMeanSource
    b -= bMeanSource
  
    # scale by the standard deviations
    l = (lStdTarget / lStdSource) * l
    a = (aStdTarget / aStdSource) * a
    b = (bStdTarget / bStdSource) * b

    # add in the source mean
    l += lMeanTarget
    a += aMeanTarget
    b += bMeanTarget

    # merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data type
    result =  cv2.merge([l, a, b])
    img_RGB = convert_color_space_CIECAM97s_to_RGB(result)
    (r,g,b) = cv2.split(img_RGB)

    r = np.clip(r, 0, 255.0)
    g = np.clip(g, 0, 255.0)
    b = np.clip(b, 0, 255.0)

    img_RGB =  cv2.merge([r, g, b])
    img_RGB = np.uint8(img_RGB)

    return img_RGB

def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW1: color transfer')
    print('==================================================')

    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]
    
    # ===== read input images
    # img_RGB_source: is the image you want to change the its color
    # img_RGB_target: is the image containing the color distribution that you want to change the
    # img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)

    img_BGR_source = cv2.imread('source1.png')
    img_BGR_target = cv2.imread('target1.png')

    img_RGB_source = convert_color_space_BGR_to_RGB(img_BGR_source)
    img_RGB_target = convert_color_space_BGR_to_RGB(img_BGR_target)

    img_RGB_new_Lab = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    img_BGR_new_Lab = convert_color_space_RGB_to_BGR(img_RGB_new_Lab)
    cv2.imwrite('output_in_Lab1.png', img_BGR_new_Lab)
    
    img_RGB_new_RGB = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    img_BGR_new_RGB = convert_color_space_RGB_to_BGR(img_RGB_new_RGB)
    cv2.imwrite('output_in_RGB1.png', img_BGR_new_RGB)

    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    img_BGR_new_CIECAM97s = convert_color_space_RGB_to_BGR(img_RGB_new_CIECAM97s)
    cv2.imwrite('output_in_CIECAM97s1.png', img_BGR_new_CIECAM97s)

