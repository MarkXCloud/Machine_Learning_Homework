import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from PCA import PCA
from K_means import K_means
from Adaboost import Adaboost
from get_feature import get_HOG

np.random.seed(234)


def pre_process(img_data, img_width, img_height):
    # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)  # opencv read images in the BGR order so we convert it
    img_data = cv2.resize(img_data, (img_width, img_height),
                          interpolation=cv2.INTER_CUBIC)  # resize the image to proper size
    img_data = np.float32(img_data)#convert into float
    return img_data


def divide_img_hog(image, step=3, pca=False):
    result = []
    for img in image:
        w, h, _ = img.shape
        for j in range(0, h, step):
            for i in range(0, w, step):
                sub_img = img[i:i + step, j:j + step]
                hog_feature = get_HOG(sub_img)
                result.append(hog_feature)
    return np.array(result)


def divide_img_label(image, step=3):
    result = []
    for img in image:
        w, h = img.shape
        for j in range(0, h, step):
            for i in range(0, w, step):
                sub_img = img[i:i + step, j:j + step]
                mode = stats.mode(sub_img.flatten())[0][0]  # we use the mode of a rigion as its label
                result.append(mode)
                #print(mode)
    return np.array(result)


def predict(img, classifyer, step=3):
    w, h, _ = img.shape
    result_img = np.zeros(shape=(w, h), dtype=np.float32)
    for j in range(0, h, step):
        for i in range(0, w, step):
            sub_img = img[i:i + step, j:j + step]
            #print(get_HOG(sub_img))
            result = classifyer.predict(np.expand_dims(get_HOG(sub_img), 0))
            if result==None:
                result=[0]
            result_img[i:i + step, j:j + step] += result[0]

    return result_img


if __name__ == '__main__':
    img_lst = [ '280',  '1560', '3287',  '100015','133391','203099','208365']
    img_path = './MSRA10K_Imgs_GT/Imgs'
    train_data = []
    train_label = []
    train_hog = []
    train_hog_label = []
    img_height = 288  # resize shape: (225x225)
    img_width = 288
    for img in img_lst:
        print("Step1: Image reading.")

        X_train = cv2.imread("./clus_Imgs/"+img+".jpg")
        X_train = cv2.resize(X_train,(img_width,img_height),interpolation=cv2.INTER_CUBIC)
        X_train = np.expand_dims(X_train,0)
        y_train = cv2.imread("./MSRA10K_Imgs_GT/Imgs/"+img+".png")
        y_train = cv2.resize(y_train,(img_width,img_height),interpolation=cv2.INTER_CUBIC)
        y_train = np.float32(y_train)
        y_train = cv2.cvtColor(y_train,cv2.COLOR_BGR2GRAY)
        y_train /= 255
        y_train[y_train>0.5] = 1
        y_train[y_train<0.5] = -1
        y_train = np.expand_dims(y_train, 0)
        print("Step3: Divide picture")
        # adaboost = Adaboost(num_classifier=5)  # Adaboost + SVM
        one_scale_data = divide_img_hog(X_train, step=img_height // 2)
        one_scale_label = divide_img_label(y_train, step=img_height // 2)
        two_scale_data = divide_img_hog(X_train, step=img_height // 3)
        two_scale_label = divide_img_label(y_train, step=img_height // 3)
        three_scale_data = divide_img_hog(X_train, step=img_height // 3 // 3)
        three_scale_label = divide_img_label(y_train, step=img_height // 3 // 3)

        print("Step4: Train Adaboost")
        print("Training Classifier No.1")
        one_scale_ada = Adaboost(num_classifier=5)
        one_scale_ada.load_data(data=one_scale_data, label=one_scale_label)
        one_scale_ada.fit()
        print("Training Classifier No.2")
        two_scale_ada = Adaboost(num_classifier=5)
        two_scale_ada.load_data(data=two_scale_data, label=two_scale_label)
        two_scale_ada.fit()
        print("Training Classifier No.3")
        three_scale_ada = Adaboost(num_classifier=5)
        three_scale_ada.load_data(data=three_scale_data, label=three_scale_label)
        three_scale_ada.fit()
        print("Step5: test")
        # Start at X_train[0]
        result1 = predict(X_train[0], classifyer=one_scale_ada, step=img_height // 2)
        result2 = predict(X_train[0], classifyer=two_scale_ada, step=img_height // 3)
        result3 = predict(X_train[0], classifyer=three_scale_ada, step=img_height // 3 // 3)
        result = (result1*2 + result2 + result3*3)/6
        result[result>0.4]=1
        result[result<0.4]=0

        result = (result*255).astype(np.uint8)


        #cv2.imshow("temp_result", result)
        cv2.imwrite("temp_result/"+img+".png", result)
    #cv2.waitKey(0)
