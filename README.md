# Machine_Learning_Homework
> ----by Zili Wang

This is my repo to commit my machine learning homework.Including:

① Support Vector Machine

② Adaboost

③ Gaussian Mixture Model

④ Hidden Markov Model

Additional homework will be submitted as long as I get all my works done. However it will not until summer vacation that I have more separate time to share more codes of my own. ILife is hard, and let's manage it.o(╥﹏╥)o


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

## 3.Details
When you runs the SVM.py, it will draw the raw data as a scatter and the draw a picture with data point and classify straight line, which is our hyper plane.
In this file we have our central class *Support_Vector_Machine()*. And to easily get hands on, I put *fit(), pridict()* method in it.

In SMO.py the central function is *smo()*. Above it are some small function like *clip() and select_j()* to complete the fundamental functions of SMO.

In Adaboost.py the *Adaboost()* is our central class. You can choose *SVM* or *Line* as your weak classifier to run and watch their performance.

The Horizontal_Classifier.py includes a single line classifier. You can imagine it as a preceptron, premitive and only has one dimention. By the way,please do not get the learning rate *alpha* to big or the line will swinging around.

The GMM.py contains an one dimentional Gaussian Mixture Model. All the parameters would be calculated in a matrix way rather than going through each of them to sum together and divide and bulabulabula... 

The HMM.py used the dynamic programming to seach the best way. However there might be some small bugs of data types in it so I still need time to fix it.
