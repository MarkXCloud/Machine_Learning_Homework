import cv2 as cv
import numpy as np

img_lst = [ '280',  '1560', '3287',  '100015','133391','203099','208365']
def clamp(pv):  #防止溢出
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv


def gaussian_noise(image):  # 获取有高斯噪声的图片
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)  #产生随机数
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0]) #产生有高斯噪声的图片
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    #cv.imshow("noise image", image)


for img in img_lst:
    print(img)
    src = cv.imread("temp_result/"+img+".png")
    #cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    #cv.imshow("input image", src)

    t1 = cv.getTickCount()
    gaussian_noise(src)
    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print("time consume : %s"%(time*1000))
    dst = cv.GaussianBlur(src, (0, 0), 15) #（0,0）,然后根据sigmaX=15自动计算ksize,高斯模糊对高斯噪声有抑制作用
    dst[dst>126]=255
    dst[dst<=126]=0
    """
    GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
    
    src，输入图像，即源图像，填Mat类的对象即可。图片深度应该为CV_8U,CV_16U, CV_16S, CV_32F 以及 CV_64F之一。
    dst，即目标图像，需要和源图片有一样的尺寸和类型。比如可以用Mat::Clone，以源图片为模板，来初始化得到如假包换的目标图。
    ksize，高斯内核的大小。其中ksize.width和ksize.height可以不同，但他们都必须为正数和奇数（并不能理解）。或者，它们可以是零的，它们都是由sigma计算而来。
    sigmaX，表示高斯核函数在X方向的的标准偏差。　　根据这个可以获取sigmaY,若是sigmaX和sigmaY都没有则根据ksize获取
    sigmaY，表示高斯核函数在Y方向的的标准偏差。若sigmaY为零，就将它设为sigmaX，如果sigmaX和sigmaY都是0，那么就由ksize.width和ksize.height计算出来。
    为了结果的正确性着想，最好是把第三个参数Size，第四个参数sigmaX和第五个参数sigmaY全部指定到。
    borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。
    """
    #cv.imshow("Gaussian Blur", dst)
    cv.imwrite("./result/"+img+".png",dst)



#cv.waitKey(0)

#cv.destroyAllWindows()
