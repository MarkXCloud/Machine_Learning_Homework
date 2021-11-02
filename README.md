# Machine_Learning_Homework
> ----by Zili Wang

This is my repo to commit my machine learning homework.Including:

1.Support Vector Machine

2.Adaboost

3.Gaussian Mixture Model

4.Hidden Markov Model

5.Principal Conponent Analysis

6.K-means that I applied some new method

7.Final work of Salient Object Detection

I now submit my final work after a lot of events in my last year of university life. I am going to finish my B.S. In the fulture I will work on pattern recognition and deep learning. My Machine Learning codes will be refreshed as long as I am avaliable to keep coding.


## 1. Requirements
numpy>=1.16.1

matplotlib

sklearn

## 2.Usage
This repo contains: Adaboost.py, Horizontal_Classify.py, SVM.py, SMO.py , GMM.py and Load_data.py.

The `Adaboost.py` includes the adaboost algorithm with some weak classifier.

The `Horizontal_Classifier.py` is a simple line classifier to make the performance of adaboost more obvious.

The `SVM.py` includes the main code to run my svm algorithm.

The `SMO.py` realizes the SMO algorithm mostly according to the textbook. However, I didn't use its method to choose α by judging the KKT condition but to try random
choose. Surprisingly it worked better than the former.

The `GMM.py` contains a GMM model.

The `HMM.py` contains the method to calculate hidden markov model.

The `Load_data.py` only contains the function of load iris dataset.

The `PCA.py` is a function of apply PCA. It was firstly designed for my final work but abandoned since it couldn't perform well on my duty.

The `get_feature.py`, `K_means.py`, `Filter.py`, `cal_IOU.py` together with Adaboost and SVM are the files used in my final work.

## 3.Details
When you runs the SVM.py, it will draw the raw data as a scatter and the draw a picture with data point and classify straight line, which is our hyper plane.
In this file we have our central class *Support_Vector_Machine()*. And to easily get hands on, I put *fit(), pridict()* method in it.

In SMO.py the central function is *smo()*. Above it are some small function like *clip() and select_j()* to complete the fundamental functions of SMO.

In Adaboost.py the *Adaboost()* is our central class. You can choose *SVM* or *Line* as your weak classifier to run and watch their performance.

The Horizontal_Classifier.py includes a single line classifier. You can imagine it as a preceptron, premitive and only has one dimention. By the way,please do not get the learning rate *alpha* to big or the line will swinging around.

The GMM.py contains an one dimentional Gaussian Mixture Model. All the parameters would be calculated in a matrix way rather than going through each of them to sum together and divide and bulabulabula... 

The HMM.py used the dynamic programming to seach the best way. However there might be some small bugs of data types in it so I still need time to fix it.

The K_means.py firstly precesses the data from https://mmcheng.net/msra10k/ . The reason for this is from a naive idea: in a picture, the foreground will have the most details while the background will have few. So I applied K-means to decrease the degree of details in a picture, with foreground kept more and background kept less. Then I perform get_feature.py to get HOG of the clu_Img. Since more details was conserved for foreground, the border of these details were more than the background, so I could get gradient information more than the background. This may be too simple but it work to some degree. After that I will use a Adaboost+SVM classifier to classify these HOG to fore or background. What's not satisfied is that my device was weak to hold image process during the work, so I used quite large slide windows around the whole work. That led to the big  bunch of white boxes in my result. To fix it, I tried to apply Gaussian filter in the Filter.py to fix the shape. Finally I got the result. It performed good and most of the time it can locate the object correctly and draw the rough shape of the foreground.Ues cal_IOU.py will evaluate the IOU between my result and GT, and the results are quite well. Really Nice!😁
