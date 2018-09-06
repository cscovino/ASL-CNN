# ASL-CNN

A translator for american sign language built with convolutional neural networks (CNN).
For this project I only took four letters of the alphabet: A, B, C and D. So, the model was trained only for this signs and it could predict only these letters.
![asl](https://user-images.githubusercontent.com/9748855/45159128-5900e900-b1bc-11e8-8d9c-60ee3115bf35.png)

For training the model I used the dataset from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet) (Thanks to Akash)
and also [this one](https://www.kaggle.com/danrasband/asl-alphabet-test) (Thanks  to Dan Rasband)

It is important to know that in the project folder you already have the model trained, if you want to create another model you have to download the datasets and then preprocess the data (you should modify the original code).

## Getting started

To run and test the app first you need a copy of the project. For this you have to run this command in the command-line

```
 git clone https://github.com/cscovino/ASL-CNN.git
```

This assumes that you have installed [git](https://git-scm.com/).

### Prerequisites

It is needed to have installed [Python 3.5.6](https://www.python.org/downloads/).

### Installing

Once you have cloned the repository you need to install some libraries of python (if you don't have them).

```
 pip install numpy
 pip install opencv-python
 pip install --upgrade tensorflow
 pip install tflearn
 pip install matplotlib.pyplot
 pip install tqdm
```

## Running the code

Within the project folder you have to run this command.
```
 python image.py
```
Then the app will ask you for an option, there are four options to chose.

![options](https://user-images.githubusercontent.com/9748855/45190600-26cfa580-b215-11e8-9399-80611b6af835.png)

a) It is for create and train the model if you don't have any model in the folder.
b) It is to test the model with images in the folder "test".
c) It is to tes the model but taking the pictures with the webcam, using opencv.

If you select c) then you have to press c to capture an image and predict the letter with your model. If you want return to main menu just press q.

![test](https://user-images.githubusercontent.com/9748855/45190512-b032a800-b214-11e8-9d54-cb44b034b537.png)

## Built with
* [Python](https://www.python.org/) - An interpreted high-level programming language.
* [OpenCV](https://opencv.org/) - An open source computer vision and machine learning software library.
* [Tensorflow](https://www.tensorflow.org/) - An open source software library for high performance numerical computation.
* [TFLearn](http://tflearn.org/) - A modular and transparent deep learning library built on top of Tensorflow.
