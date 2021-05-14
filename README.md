# Machine_Learning_Homework
This is my repo to commit my machine learning homework.
#Homework On Support Vector Machine
---------by Zili Wang

## 1. Requirements
numpy>=1.16.1

matplotlib

sklearn

## 2.Usage
There are the py files in my homework:SVM.py, SMO.py and Load_data.py.

The SVM.py includes the main code to run my svm algorithm.

The SMO.py realizes the SMO algorithm mostly according to the textbook. However, I didn't use its method to choose α by judging the KKT condition but to try random
choose. Surprisingly it worked better than the former.

The Load_data.py only contains the function of load iris dataset.

## 3.Details

When you runs the SVM.py, it will draw the raw data as a scatter and the draw a picture with data point and classify straight line, which is our hyper plane.
In this file we have our central class *Support_Vector_Machine()*. And to easily get hands on, I put *fit(), pridict()* method in it.

In SMO.py the central function is *smo()*. Above it are some small function like *clip() and select_j()* to complete the fundamental functions of SMO.
