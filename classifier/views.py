from django.shortcuts import render
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imread, imresize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys,os
from keras.models import model_from_json
import tensorflow as tf

# Create your views here.
def home(request):
    return render(request,"index.html")

#tell our app where our saved model is
sys.path.append(os.path.abspath("./"))

def init():
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return loaded_model,graph


def predict(request):
    class_table = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
    ]
    if request.method == 'POST': # If the form has been submitted...
        form = request.POST # A form bound to the POST data
        
        f=request.FILES['fileToUpload'] # Getting the image uploaded from the UI as f
        #saving files to profile_image/image1.jpg
        with open('output.png','wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
    print("debug")
	#read the image into memory
    x = imread('output.png',mode='L')
    #compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
	#make it the right size
    x = imresize(x,(28,28))
    #imshow(x)
	#convert to a 4D tensor to feed into our model
    x = x.reshape(1,28,28,1)
    model, graph = init()
    print("debug2")
    #in our computation graph
    with graph.as_default():
        #perform the prediction
        out = model.predict(x)
        print(out)
        print(np.argmax(out,axis=1))
        print("debug3")
        #convert the response to a string
        response = np.array_str(np.argmax(out,axis=1))
    print(class_table[int(response)])
    return render(request,"result.html",{"result":response})



