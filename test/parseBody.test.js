const parseBody = require('../src/lib/parseBody')
const sanitizeHTML = require('../src/lib/sanitizeHTML')
const html = `
Today we are going to develop a computer vision application for detecting if the eyes are open or close and count blinks. To achieve our goal we are going to train a small convolutional neural network (CNN) with Keras and then using OpenCV and dlib we 'll implement our blink detector .

The process of building our blink detector has two stages, first the training of the neural network and then the development of the detector. If you want you can skip stage one and go to stage two and use the network that we have already trained for you. You can find the code at my <a href="https://github.com/iparaskev/simple-blink-detector">github repo</a>.
<h3><strong>Stage one </strong></h3>
For these stage we 'll assume that you already know a fair bit about convolutional neural nets. That's because we won't talk about how cnn's work. If these is the first time you hear about cnn's then you can go straight to stage two. Stanford has a<a href="http://cs231n.github.io/"> great course</a> which will help you understand a lot about neural networks and cnn's .

To get started lets see what we need to train our cnn. We 'll use keras library with tensorflow backend, keras gives you the choise to use it with tensorflow or theano backend. Also you can use either python 2.7 or python 3.x . If you don't have keras or tensorflow at your system you can
<pre class="lang:sh decode:true">$ pip install --upgrade tensorflow</pre>
and
<pre class="lang:default decode:true">$ pip install keras</pre>
, tensorflow supports CUDA if you have CUDA-capable GPU but for these tutorial it won't make a huge difference.

So we are going to train a binary classifier between open and close eyes. To do this we 'll need a dataset to train our classifier. For closed eyes we will use cropped images of size 26x34 from the <a href="http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html">Closed Eyes In The Wild (CEW)</a> dataset and for opened eyes we used a manually anotaded images. We are going to use only left eye images because our dataset is small and we want the cnn to be more accurate, to achieve that we flipped the right images when we were cropping the whole face images. The complete dataset contains 2874 images. You can find the dataset in at csv format in the train folder in my repo.

To get started we 'll import the necessary packages:
<pre class="lang:python decode:true">import csv
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.activations import relu
from keras.optimizers import Adam</pre>
Now that we have imported everything we need to load the data from the dataset, to do that we are going to write a function
<pre class="tab-convert:true lang:default decode:true">def readCsv(path):

	with open(path,'r') as f:
		#read the scv file with the dictionary format
		reader = csv.DictReader(f)
		rows = list(reader)

	#imgs is a numpy array with all the images
	#tgs is a numpy array with the tags of the images
	imgs = np.empty((len(list(rows)),height,width,1),dtype=np.uint8)
	tgs = np.empty((len(list(rows)),1))

	for row,i in zip(rows,range(len(rows))):

		#convert the list back to the image format
		img = row['image']
		img = img.strip('[').strip(']').split(', ')
		im = np.array(img,dtype=np.uint8)
		im = im.reshape((26,34))
		im = np.expand_dims(im, axis=2)
		imgs[i] = im

		#the tag for open is 1 and for close is 0
		tag = row['state']
		if tag == 'open':
			tgs[i] = 1
		else:
			tgs[i] = 0

	#shuffle the dataset
	index = np.random.permutation(imgs.shape[0])
	imgs = imgs[index]
	tgs = tgs[index]

	#return images and their respective tags
	return imgs,tgs</pre>
This function accepts a single required parameter, the path of the csv file with the dataset.

At first we read the csv file with the dictionary format and then we make a list with every row of the file. Then we make two empty numpy arrays to store the images and the tag of every image. After that we access through every row of the list, which contains an image and the image's tag , to assert to the previous arrays their values. In the end we shuffle the two arrays and we return them.

So to continue we are going to build our cnn using keras. Our network has three convolutional filters with relu activation, each filter followed by a max-pooling layer. Then we add dropout a dropout layer followed by two fully connected layers with relu activations also. Finally we add a single neuron with sigmoid activation for our binary classifier. As optimizer we 'll use adam and for our loss function we 'll use binary crossentropy.
<pre class="lang:default decode:true">#make the convolution neural network
def makeModel():
	model = Sequential()

	model.add(Conv2D(32, (3,3), padding = 'same',
                   input_shape=(height,width,1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (2,2), padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (2,2), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))


	model.compile(optimizer=Adam(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

	return model</pre>
Now we have all we need to train our small and simple cnn.
<pre class="lang:default decode:true">def main():

	xTrain ,yTrain = readCsv('dataset.csv')

	#scale the values of the images between 0 and 1
	xTrain = xTrain.astype('float32')
	xTrain /= 255

	model = makeModel()

	#do some data augmentation
	datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        )
	datagen.fit(xTrain)

	#train the model
	model.fit_generator(datagen.flow(xTrain,yTrain,batch_size=32),
			    steps_per_epoch=len(xTrain) / 32, epochs=50)

	#save the model
	model.save('blinkModel.hdf5')</pre>
&nbsp;

First we load our images and the tags at two numpy arrays, then we scale the values of the images between 0 and 1, we do that because it makes the learning process faster. After that we do some data augmentation at our data to artificially increase the number of the training examples, because we have a small dataset and we have to reduce overfitting. Finally we train out network for 50 epochs with batch size of 32 and we save our trained cnn.

We know that normally we had to split our data at train, val and test sets, do some fine tuning and then train our network to evaluate it at our test set, but the purpose of this tutorial is to make fast a simple blink detector and not how to train a cnn classifier for open and close eyes. So to achieve that we don't give a lot of importance to really important steps of the training.
<h3 id="stage-two">Stage two</h3>
Now we have our trained cnn and we are ready  to build our blink detector.  Lets see what libraries we 'll need.
<pre class="lang:default decode:true ">import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils</pre>
The computer vision library we are going to use is OpenCV, if you don't have it you can install it following the instructions given <a href="https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/">here</a> for Ubuntu 16.04. Also we 'll use dlib and for a set of convenience functions to make working with OpenCV easier we 'll need to use imutils library. If you don't have any of those two installed on your system you can install them easily using
<pre class="lang:default decode:true ">$ pip install --upgrade imutils</pre>
and for dlib you can follow <a href="https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/">this guide</a> .

Overview of the detector: First we read each frame from the camera, then we crop the eyes and we give them to the cnn we have trained to make a prediction on them. After that we take the mean of the predictions because we look for blinks so we have to be sure. In the end we counter the consecutive close predictions and if they are more than a threshold we count it as a blink. Lets see some code.

Now we 'll define a function for face detection. We 'll use haarcascade's face detector because is faster than dlib's frontal face detector.
<pre class="lang:default decode:true ">face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# detect the face rectangle
def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)

    # if it doesn't return rectangle return array
    # with zero lenght
    if len(rects) == 0:
        return []

    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    return rects</pre>
This function accepts a single required parameter the whole frame.

At <strong>first line</strong> we load the haarcascede classifier from the xml file, which you can find in the repo.

<strong>Lines 5-7 </strong>we check if the classifier has correctly loaded.

<strong>Lines 11-12 </strong>we check if the classifier hasn't find a rectangle with the face and return an empty list if it didn't.

Finally at <strong>line 15</strong> we convert the rectangle list from [x,y,a,b] where (x,y) are the coordinates of the left corner of the rectangle and a,b are the pixels we have to add to x and y respectively to form the whole rectangle, to [x,y,maxX,maxY]. Then we return the rectangle list which contains zero, one or more rectangles.

Now that we have find frame's rectangle  which contains the face, we can proceed to find the eyes.  Lets make a function for this.
<pre class="lang:default decode:true">predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def cropEyes(frame):

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect the face at grayscale image
	te = detect(gray, minimumFeatureSize=(80, 80))

	# if the face detector doesn't detect face
	# return None, else if detects more than one faces
	# keep the bigger and if it is only one keep one dim
	if len(te) == 0:
		return None
	elif len(te) &gt; 1:
		face = te[0]
	elif len(te) == 1:
		[face] = te</pre>
Also these function accepts one required parameter the frame.

At <strong>line 1 </strong>we initialize the dlib's face predictor. You can learn more about it in <a href="https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/">this blog post</a>.

On <strong>line 4 </strong>we convert our frame to greyscale. Then on <strong>lines 6-17 </strong>we assign at te the value of the face detect function and then we check if it is empty and return none if it is, because we don't want the display of our predictor to stop(it will become more clear in a few minutes).
<pre class="lang:default decode:true"># keep the face region from the whole frame
	face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
								right = int(face[2]), bottom = int(face[3]))

	# determine the facial landmarks for the face region
	shape = predictor(gray, face_rect)
	shape = face_utils.shape_to_np(shape)

	#  grab the indexes of the facial landmarks for the left and
	#  right eye, respectively
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# extract the left and right eye coordinates
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]</pre>
First we on <strong>line 2</strong> we take the face region from the whole frame and then we determine the facial landmarks for the face region, while <strong>line </strong><strong>5</strong> converts these coordinates to NumPy array.

<strong>Lines </strong><strong>11-12 </strong>we grab the indexes of the facial landmarks for the left and right eye from the full set of dlib's facial landmarks.

Next we extract the left and right eye coordinates using array slicing techniques using the indexes we just had grabbed.
<pre class="lang:default decode:true "># keep the upper and the lower limit of the eye
	# and compute the height
	l_uppery = min(leftEye[1:3,1])
	l_lowy = max(leftEye[4:,1])
	l_dify = abs(l_uppery - l_lowy)

	# compute the width of the eye
	lw = (leftEye[3][0] - leftEye[0][0])

	# we want the image for the cnn to be (26,34)
	# so we add the half of the difference at x and y
	# axis from the width at height respectively left-right
	# and up-down
	minxl = (leftEye[0][0] - ((34-lw)/2))
	maxxl = (leftEye[3][0] + ((34-lw)/2))
	minyl = (l_uppery - ((26-l_dify)/2))
	maxyl = (l_lowy + ((26-l_dify)/2))

	# crop the eye rectangle from the frame
	left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
	left_eye_rect = left_eye_rect.astype(int)
	left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]</pre>
Our cnn to be able to predict on an image, the image has to be the same format as the images which it trained with. So we need to make same adjustments at the eye coordinates we have.

First on <strong>lines 3-8 </strong>we find the minimum and maximum y value from our coordinates and we compute the height of the left eye. Dlib gives 6 pairs of coordinates for the eyes,

<img class="alignnone size-medium wp-image-79" src="http://alexoglou.webpages.auth.gr/wordpress/wp-content/uploads/2017/09/dlibpost-300x300.jpg" alt="" width="300" height="300" />

as you can see we want the minimum y from the second and third pair and the maximum y from the fifth and sixth to compute eye's height. The width is much easier because we simple have to take the x from the first and the fourth pair and compute their difference.

That is what we do at <strong>line 11.</strong>

Then on <strong>lines 17-21 </strong>to compute the coordinates of the eye's rectangle we add the half of the differences of the shape we want our image to be with the width and height of the eye we have to our x and y coordinates.

After at <strong>lines 24-26 </strong>we crop from the whole image the eye rectangle.

This was for the left eye, now we 'll do the same for the right.
<pre class="lang:default decode:true "># same as left eye at right eye
	r_uppery = min(rightEye[1:3,1])
	r_lowy = max(rightEye[4:,1])
	r_dify = abs(r_uppery - r_lowy)
	rw = (rightEye[3][0] - rightEye[0][0])
	minxr = (rightEye[0][0]-((34-rw)/2))
	maxxr = (rightEye[3][0] + ((34-rw)/2))
	minyr = (r_uppery - ((26-r_dify)/2))
	maxyr = (r_lowy + ((26-r_dify)/2))
	right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
	right_eye_rect = right_eye_rect.astype(int)
	right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]</pre>
To finish our function
<pre class="lang:default decode:true "># if it doesn't detect left or right eye return None
	if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
		return None
	# resize for the conv net
	left_eye_image = cv2.resize(left_eye_image, (34, 26))
	right_eye_image = cv2.resize(right_eye_image, (34, 26))
	right_eye_image = cv2.flip(right_eye_image, 1)
	# return left and right eye
	return left_eye_image, right_eye_image</pre>
we check if we haven't detect left or right eye so we can return none and if we have detected both of the eyes we resize them to be sure the images are the right size and we before we return the eye images we flip the right eye so we can have to left for right predictions.

Before we go to the main function of our script we 'll write a function for the rest preprocess of the every image we have to do for our cnn.
<pre class="lang:default decode:true "># make the image to have the same format as at training
def cnnPreprocess(img):
	img = img.astype('float32')
	img /= 255
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	return img</pre>
Here we scale the values of the images between 0 and 1 and we add two more dimensions because keras need the image to be of shape (rows,width,height,channels) where row are the number of the images, width and height of the images and channel the number of colors. So we have one image per time and 1 channel and that is what <strong>lines 5-6 </strong>do.

Finally we are ready for our main function.
<pre class="lang:default decode:true ">def main():
	# open the camera,load the cnn model
	camera = cv2.VideoCapture(0)
	model = load_model('blinkModel.hdf5')

	# blinks is the number of total blinks ,close_counter
	# the counter for consecutive close predictions
	# and mem_counter the counter of the previous loop
	close_counter = blinks = mem_counter= 0
	state = ''
	while True:

		ret, frame = camera.read()

		# detect eyes
		eyes = cropEyes(frame)
		if eyes is None:
			continue
		else:
			left_eye,right_eye = eyes

		# average the predictions of the two eyes
		prediction = (model.predict(cnnPreprocess(left_eye)) + model.predict(cnnPreprocess(right_eye)))/2.0
</pre>
First we open our camera and we load our cnn model and we define some usable variable we 'll use and explain in a minute. Then we start reading consecutive frames.

At <strong>lines 16-19 </strong> we call our function for detecting and cropping the eyes from the whole frame and we check if the value of them is none. If it is none the script will stop so to avoid that we check for the value and we continue to the next loop if it is. After we just average our predictions for the eyes.
<pre class="lang:default decode:true "># blinks
		# if the eyes are open reset the counter for close eyes
		if prediction &gt; 0.5 :
			state = 'open'
			close_counter = 0
		else:
			state = 'close'
			close_counter += 1

		# if the eyes are open and previousle were closed
		# for sufficient number of frames then increcement
		# the total blinks
		if state == 'open' and mem_counter &gt; 1:
			blinks += 1
		# keep the counter for the next loop
		mem_counter = close_counter

		# draw the total number of blinks on the frame along with
		# the state for the frame
		cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "State: {}".format(state), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# show the frame
		cv2.imshow('blinks counter', frame)
		key = cv2.waitKey(1) &amp; 0xFF

		# if the \`q\` key was pressed, break from the loop
		if key == ord('q'):
			break</pre>
&nbsp;

<strong>Lines 3-8 </strong>we check the value of the prediction if it is more than 0.5, because we have a binary classifier with a sigmoid neuron that gives as output the probability of an eye to be open and if it more than 50 % we classify it as open. If it is open we set the state variable open and the close_counter zero. if it close then we add one to the counter so we can know for how many consecutive frames the eyes were closed.

At <strong>lines 13-14 </strong>we see if the eyes are open and memory counter variable is more than one, memory counter has the value of the close counter of the previous loop. So we can check if the eyes are open and they were previously closed for sufficient frames. At this point you can adjust the number of consecutive frames at your camera.

Then we draw on our frame the number of total blinks and the current state.

Finally we display the frame and you can stop the counter pressing the button q.
<pre class="lang:default decode:true ">	# do a little clean up
	cv2.destroyAllWindows()
	del(camera)</pre>
And here we do a little clean up.

Now you have your blink detector. It was easy to built it and fast.You can use your own dataset if you want and you can change everything you want.

&nbsp;
`

const res = parseBody(sanitizeHTML(html))

console.log(res)