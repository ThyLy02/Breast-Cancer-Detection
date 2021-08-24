# Breast Cancer Detection project
 

## Introduction

Breast Cancer Dectection by using Deep Neural Network is an Artificial Intelligent project. 
It is a small project in class for practicing some kind of models in Machine Learning.
My model is still simple, but I will reorganize it better in next time.


## Installation

**Create environment**

``` bash
conda create -n myenv python=3.8.3
conda activate myenv
```

**Install package**

``` bash
pip install -r requirements.txt
```

**Download and information of dataset**

Go to folder data, there are 4 files (xtest, ytest, xtrain, ytrain).
The Breast Cancer Wisconsin Dataset have 569 samples, 31 index.
Sorting it into training and test sets with the 'input' values to the Neural Network as 'X' values
The expected 'output' (a 0 if benign and a 1 if malignant) as the 'Y' values.
There are 2 classes, 4 layers.


## Usage

**How to run**

``` bash
python train.py
```

**Expected output**

![expected output training image](https://github.com/ThyLy02/Breast-Cancer-Detection/blob/main/images/trainimage.png)

**After the model is trained, run the test file:**

``` bash
python test.py
```



