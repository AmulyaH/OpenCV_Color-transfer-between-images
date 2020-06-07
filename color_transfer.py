import cv2
import numpy as np
import sys
import math

MRGB_to_XYZ = np.array([[0.5141, 0.3239, 0.1604],
                        [0.2651, 0.6702, 0.0641],
                        [0.0241, 0.1228, 0.8444]])

MXYZ_to_LMS = np.array([[0.3897, 0.6890, -0.0787],
                        [-0.2298, 1.1834, 0.0464],
                        [0.0000, 0.0000, 1.0000]])

MRGB_to_LMS =  np.matmul( MXYZ_to_LMS , MRGB_to_XYZ )

Matrix_RGB_To_LMS = np.array([[0.3811, 0.5783, 0.0402],
                              [0.1967, 0.7244, 0.0782],
                              [0.0241, 0.1288, 0.8444]])

Matrix_LMS_To_LAB = np.matmul(np.array([[1/math.sqrt(3), 0, 0], 
                              [0,1/math.sqrt(6),0], 
                               [0, 0, 1/math.sqrt(2)]]) , np.array([[1,1, 1],[1,1,-2],[1, -1,0]]))

TMatrix_LMS_To_RGB = np.array([[4.4679, -3.5873, 0.1193],
                              [-1.2186, 2.3809, -0.1624],
                              [0.0497, -0.2439, 1.2045]])


Matrix_LAB_To_LMS = np.matmul( np.array([[1,1, 1],[1,1,-1],[1, -2,0]]) , np.array([[math.sqrt(3)/3, 0, 0], [0,math.sqrt(6)/6,0],  [0, 0, math.sqrt(2)/2]]) )
                              

#TMatrix_LMS_To_RGB = np.linalg.inv(Matrix_RGB_To_LMS)

MLMS_to_LAB = np.array([[1/(np.sqrt(3)), 0, 0],
                              [0, 1/(np.sqrt(6)), 0],
                              [0, 0, 1/(np.sqrt(2))]])

M2LMS_to_LAB = np.array([[1, 1, 1],
                        [1, 1,-2],
                        [1, -1, 0]])

MLMS_to_CIECAM97s = np.array([[2.00, 1.00, 0.05],
                            [1.00, -1.09, 0.09],
                            [0.11, 0.11, -0.22]])

TMatrix_CIECAM97s_to_LMS = np.linalg.inv(MLMS_to_CIECAM97s)

'''
    img_Lab = convert_color_space_RGB_to_Lab(img_RGB_source)
    img_BGR = convert_color_space_RGB_to_BGR(img_RGB_source)
    img_RGB = convert_color_space_BGR_to_RGB(img_BGR)
    img_LAB_RGB = convert_color_space_Lab_to_RGB(img_Lab)
    img_CIECAM97s = convert_color_space_RGB_to_CIECAM97s(img_RGB)
    img_RGB = convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s)
    print(img_RGB)
'''

def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR,dtype=np.float32)
    img_RGB[:,:,0] = img_BGR[:,:,2]
    img_RGB[:,:,2] = img_BGR[:,:,0]
    img_RGB[:,:,1] = img_BGR[:,:,1]

    # to be completed ...
    return img_RGB

def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB,dtype=np.float32)
    img_BGR[:,:,0] = img_RGB[:,:,2]
    img_BGR[:,:,2] = img_RGB[:,:,0]
    img_BGR[:,:,1] = img_RGB[:,:,1]
    # to be completed ...
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
            img_LMS[r,c,:] = np.dot(MRGB_to_LMS , p) ##np.log  (Matrix_RGB_To_LMS2 @ 
            img_LMS[r,c,:] = np.log10(img_LMS[r,c,:])
            img_Lab[r,c,:] = np.dot(Matrix_LMS_To_LAB , img_LMS[r,c,:]) 

    cv2.imshow('rgb_to_lab', img_Lab.astype("uint8"))
    #cv2.waitKey(0)
        
    '''
    img_RGB_reshaped = img_RGB.transpose(2,0,1).reshape(3,-1)
    img_LMS = Matrix_RGB_To_LMS @ img_RGB_reshaped
    img_LMS = img_LMS.reshape(z, x, y).transpose(1, 2, 0).astype(np.uint8)

    img_LMS = np.log(img_LMS)
   '''

    # to be completed ...
 
    '''
    img_LMS_reshaped = img_LMS.transpose(2,0,1).reshape(3,-1)
    img_Lab = MLMS_to_LAB @ M2LMS_to_LAB @ img_LMS_reshaped
    img_Lab = img_Lab.reshape(z, x, y).transpose(1, 2, 0)

    for r in range(img_LMS.shape[0]-1):
        for c in range(img_LMS.shape[1]-1):
            p = img_LMS[r,c,:]
            p = np.array(p)
            img_Lab[r,c,:] = TMatrix_LMS_To_RGB @ p 
    
    '''
    
    # to be completed ...
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
            img_LMS[r,c,:] = np.dot(Matrix_LAB_To_LMS , p)#np.power(, 10) # np.power
            img_LMS[r,c,:] = np.power(10, img_LMS[r,c,:] )
            img_RGB[r,c,:] = np.dot(TMatrix_LMS_To_RGB , img_LMS[r,c,:] )

    cv2.imshow('lab_to_Rgb', np.ceil(img_RGB).astype("uint8"))
    #cv2.waitKey(0)    

    '''
    img_Lab_reshaped = img_Lab.transpose(2,0,1).reshape(3,-1)
    M1Lab_to_LMS = np.transpose(MLMS_to_LAB)
    M2Lab_to_LMS = np.transpose(M2LMS_to_LAB)

    img_LMS =  M2Lab_to_LMS @ M1Lab_to_LMS @ img_Lab_reshaped

    # to be completed ...
    

    img_RGB = TMatrix_LMS_To_RGB @ img_LMS

    img_RGB = img_RGB.reshape(z, x, y).transpose(1, 2, 0).astype(np.uint8)
    '''

    # to be completed ...

    return img_RGB

def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    img_CIECAM97s = np.zeros_like(img_RGB,dtype=np.float32)
    img_LMS = np.zeros_like(img_RGB,dtype=np.float32)
    x,y,z = img_RGB.shape
    img_RGB_reshaped = img_RGB.transpose(2,0,1).reshape(3,-1)
    img_LMS = Matrix_RGB_To_LMS @ img_RGB_reshaped

    img_CIECAM97s = MLMS_to_CIECAM97s @ img_LMS

    img_CIECAM97s = img_CIECAM97s.reshape(z, x, y).transpose(1, 2, 0).astype(np.uint8)

    # to be completed ...

    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_RGB = np.zeros_like(img_CIECAM97s,dtype=np.float32)
    x,y,z = img_CIECAM97s.shape
    img_CIECAM97s_reshaped = img_CIECAM97s.transpose(2,0,1).reshape(3,-1)
    img_LMS = TMatrix_CIECAM97s_to_LMS @ img_CIECAM97s_reshaped

    img_RGB = TMatrix_LMS_To_RGB @ img_LMS

    img_RGB = img_RGB.reshape(z, x, y).transpose(1, 2, 0).astype(np.uint8)

    # to be completed ...

    return img_RGB

def image_stats(img_Lab):
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(img_Lab)
  
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())


	# return the color statistics
	return (lMean, l.std(), aMean, a.std(), bMean, b.std())

def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    img_Lab_Source = convert_color_space_RGB_to_Lab(img_RGB_source)
    img_Lab_Target = convert_color_space_RGB_to_Lab(img_RGB_target)

    #img_Lab_Source = cv2.cvtColor(img_RGB_source, cv2.COLOR_RGB2LAB).astype("float32")
    #img_Lab_Target = cv2.cvtColor(img_RGB_target, cv2.COLOR_RGB2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSource, lStdSource, aMeanSource, aStdSource, bMeanSource, bStdSource) = image_stats(img_Lab_Source)
    (lMeanTarget, lStdTarget, aMeanTarget, aStdTarget, bMeanTarget, bStdTarget) = image_stats(img_Lab_Target)

    '''
    # subtract the means from the source image
    (l, a, b) = cv2.split(img_Lab_Source)
    l -= lMeanSource
    a -= aMeanSource
    b -= bMeanSource '''

    (l, a, b) = cv2.split(img_Lab_Source)
    l -= lMeanSource
    a -= aMeanSource
    b -= bMeanSource


    # scale by the standard deviations
    l = l * (lStdTarget / lStdSource ) 
    a = a * (aStdTarget / aStdSource ) 
    b = b * (bStdTarget / bStdSource ) 

    '''
    # scale by the standard deviations
    l = (lStdSource / lStdTarget) * l
    a = (aStdSource / aStdTarget) * a
    b = (bStdSource / bStdTarget) * b 
    '''


    # add in the source mean
    l += lMeanTarget
    a += aMeanTarget
    b += bMeanTarget

    '''
    # clip the pixel intensities to [0, 255] if they fall outside
	# this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255) '''

    # merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
    result =  cv2.merge([l, a, b])#img_Lab_Source 
    img_RGB = convert_color_space_Lab_to_RGB(result)
    img_RGB = (img_RGB).astype("uint8")
    #img_RGB = transfer = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_LAB2RGB)
    #img_RGB = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_LAB2RGB)
    # to be completed ...

    return img_RGB

def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    # to be completed ...

def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    # to be completed ...

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

    '''
    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5] 
    '''

    # ===== read input images
    # img_RGB_source: is the image you want to change the its color
    # img_RGB_target: is the image containing the color distribution that you want to change the
    # img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)

    img_RGB_source = cv2.imread('source1.png',cv2.IMREAD_UNCHANGED)
    img_RGB_target = cv2.imread('target1.png',cv2.IMREAD_UNCHANGED)

    img_RGB_source = img_RGB_source[:,:,0:3]
    img_RGB_target = img_RGB_target[:,:,0:3]
    #img_RGB_source = cv2.cvtColor(path_file_image_source, cv2.COLOR_BGR2RGB)
    #img_RGB_target = cv2.cvtColor(path_file_image_target, cv2.COLOR_BGR2RGB)

    #img_RGB_source = convert_color_space_BGR_to_RGB(path_file_image_source)
    #img_RGB_target = convert_color_space_BGR_to_RGB(path_file_image_target)
    '''
    cv2.imshow('Source', img_RGB_source)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    cv2.imshow('Target', img_RGB_target)
    cv2.waitKey(0)
    cv2.destroyAllWindows() '''

    img_RGB_new_Lab       = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    #img_BGR_new_Lab = cv2.cvtColor(img_RGB_new_Lab, cv2.COLOR_RGB2BGR) 
    cv2.imshow('Output', img_RGB_new_Lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # todo: save image to path_file_image_result_in_Lab

    #img_RGB_new_RGB       = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    # todo: save image to path_file_image_result_in_RGB

    #img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    # todo: save image to path_file_image_result_in_CIECAM97s

