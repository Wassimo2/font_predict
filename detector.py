from os import listdir
from os.path import isfile, join
import os
import numpy
import cv2
# import matplotlib.pyplot as plt
# import pickle
# import time
# import glob
import pandas as pd

# input image dimensions
from keras.models import model_from_json

imgSize = [70,70]
input_shape = (imgSize[0],imgSize[1],1)
numpy.random.seed(123)  # for reproducibility

def read_model():
    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model

def process_income_image(input_data):
    input_data = numpy.array(input_data, dtype=numpy.float32)
    # train_data = train_data.transpose((0, 3, 1, 2))
    input_data = input_data.astype('float32')
    input_data /= 255
    # print('Train shape:', input_data.shape)
    # print(input_data.shape[0], 'train samples')
    input_data = input_data.reshape(input_data.shape[0],input_data.shape[1],input_data.shape[2],1)
    # print(input_data.shape)
    return input_data

from sklearn.preprocessing import LabelEncoder

encoder_df = pd.read_csv('labelencoder.csv', index_col=None)
# print('encoder sizes:', encoder_df.shape)
labels = encoder_df.iloc[0:encoder_df.shape[0],1:2]
labels_value = labels.values
encoder = LabelEncoder()
encoder.fit(numpy.ravel(labels_value))
# print("new:", list(new.classes_))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

def my_custom5_cnn_model(kernelno):
    model = Sequential()
    #model.add(SpatialTransformer(localization_net=locnet, output_size=(80, 80), input_shape=input_shape))
    #model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
    model.add(Conv2D(kernelno, (3, 3), input_shape=(imgSize[0],imgSize[1],1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # print (model.output_shape)
    # (None, 26, 26, 32)
    model.add(Conv2D(kernelno, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(kernelno, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(kernelno, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #model.add(Dropout(0.25))
    model.add(Conv2D(kernelno, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

#     model.add(Conv2D(256, (3, 3)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(256, (1, 1),name = "conv4"))
#     model.add(Activation('relu'))
#     model.add(Conv2D(256, (1, 1)))
#     model.add(Activation('relu'))
#     print (model.output_shape)

    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(39, activation='softmax'))

    # load weights
    model.load_weights("backups/weights0810_f1.best-2.hdf5")

    from keras import optimizers
    ad10e4 = optimizers.Adam(lr=0.00005)
    #commpile modelPython
    model.compile(optimizer=ad10e4,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = my_custom5_cnn_model(150)#my_cnn_model() # my_deeper_cnn_model()
# model.summary()

def load_predict(img_path):
    # open images in one letter folder
    mypath=img_path
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    # print(onlyfiles)
    #.Dash_store
    biasCount = 0
    for i in range(0, len(onlyfiles)):
        # print("debugging:",i, onlyfiles[i-biasCount])
        if onlyfiles[i-biasCount][0] == '.':
            del onlyfiles[i-biasCount]
            biasCount += 1
            print("after folders:",onlyfiles)
    images = numpy.empty([len(onlyfiles),imgSize[0],imgSize[1]], dtype=object)
    labelsStr = numpy.empty(len(onlyfiles), dtype=object)
    # print(images.shape)
    for n in range(0, len(onlyfiles)):
        img = cv2.imread( join(mypath,onlyfiles[n]) )
        img = cv2.resize(img,(imgSize[0], imgSize[1]))
        #convert to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply dilation and erosion to remove some noise
        kernel = numpy.ones((1, 1), numpy.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        #  Apply threshold to get image with only black and white
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        # threshold make the picture a little bumpy

        images[n] = img
        labelsStr[n] = onlyfiles[n].split(".png")[0]
        # get the font only
        # labels[n] = onlyfiles[n].split("-")[1].split(".png")[0]
        # get the font and tag
        # labels[n] = onlyfiles[n].split(".png")[0]
        # print("debugging:",labels[n])
        #
        # if (n==0):
        #     plt.subplot(330 + (n+1))
        #     plt.imshow(img, cmap=plt.get_cmap('gray'))
        #     plt.title(labels[n])
        #     plt.savefig("test.png",bbox_inches='tight')

    return [images,labelsStr]

#X_test, y_test = load_train('test_f')#('testImages')
X_test, yStr = load_predict('predict')
X_test = process_income_image(X_test)
# imgTest = X_test.reshape(X_test.shape[0], imgSize[0], imgSize[1]，1)
# print(X_test.shape, yStr.shape)

# get top n classes predict results

topn = 5

rst = model.predict(X_test, verbose=0)
topncl = [numpy.zeros(topn)]
for eachc in rst:
    k = sorted(range(len(eachc)), key=lambda x: eachc[x], reverse=True)[:topn]
    topncl = numpy.concatenate((topncl, [k]), axis=0)
topncl = numpy.delete(topncl, 0, axis=0)#imageArr[0], axis=0)
# print("top n classes:", topncl)
rst1 = model.predict_classes(X_test[0:5], verbose=1)
#print("classes:", rst1)
probs = numpy.zeros((rst.shape[0], topn))
ncount = 0
#print("probs sizes:", probs.shape, probs[0][0], topncl.shape)
for prob in probs:
    for icount in range(0, topn):
        theno = int(topncl[ncount][icount])
        prob[icount] = float(rst[int(ncount)][theno])

    ncount+=1
# print("top n probs:", probs)
rststr = numpy.zeros((rst.shape[0], topn), dtype=object)
for ncount in range(0,rststr.shape[0]):
    for icount in range(0, topn):
        thenumber = int(topncl[ncount][icount])
        thestr = encoder.inverse_transform(thenumber)
        rststr[ncount][icount] = thestr
print("top class str:",rststr)
#rststr = encoder.inverse_transform(rst)
# rst_df = pd.DataFrame({"id": list(range(1,0+len(rst)+1)),
#                           "label": rststr[0]})
# rst_df.to_csv('rstpreditct0806_testing2_50_80.csv', index=False, header=True)

# print (rststr.shape, probs.shape)
# print (rststr[0:,0])

rst_df = pd.DataFrame({"id": list(range(1,0+len(yStr)+1)),
                       "filename": yStr,
                       "p1":rststr[0:,0],
                       "p1prob":probs[0:,0],
                       "p2":rststr[0:,1],
                       "p2prob":probs[0:,1],
                       "p3":rststr[0:,2],
                       "p3prob":probs[0:,2],
                       "p4":rststr[0:,3],
                       "p4prob":probs[0:,3],
                       "p5":rststr[0:,4],
                       "p5prob":probs[0:,4]
                      })
rst_df.to_csv('preditction.csv', index=False, header=True)

print("predict finished, see preditction.csv for details.")
