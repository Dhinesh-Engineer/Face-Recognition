# Face-Recognition

Recognize and manipulate faces from Python or from the command line with
the world’s simplest face recognition library.
Built using dlib’s state-of-the-art face recognition
built with deep learning. The model has an accuracy of 99.38% on the
Labeled Faces in the Wild benchmark.
This also provides a simple face_recognition command line tool that lets
you do face recognition on a folder of images from the command line!

Face Recognition Python is the latest trend in Machine Learning techniques. OpenCV, the most popular library for computer vision, provides bindings for Python. OpenCV uses machine learning algorithms to search for faces within a picture.

Once you have a basic understanding of facial recognition using Python, you can delve deeper into the cascade of classifiers for advanced techniques in facial recognition using Python. You will also gain knowledge about the popular libraries for facial recognition using Python.

## Facial Recognition using Python Libraries
The most popular and probably the simplest way to detect faces using Python is by using the OpenCV package. Originally written in C/C++, OpenCV now provides bindings for Python.

It uses machine learning algorithms to search for faces within a picture. Faces are very complicated, made of thousands of small patterns and features that must be matched. The face recognition algorithms break the task of identifying the face into thousands of smaller, bite-sized tasks, each of which is easy to solve, known as classifiers.

A face may have 5000 or more classifiers, all of which must match for a face to be detected. Since there are at least 5,000 or more tests per block, you might have millions of calculations to do, which makes it a difficult process. To solve this, OpenCV uses cascades.

The OpenCV cascade breaks the problem of detecting faces into multiple stages. It performs a detailed test for each block. The algorithm may have 30 to 50 of these stages or cascades, and it will only detect a face if all stages pass.

The cascades are a bunch of XML files that contain OpenCV data used to detect objects. You initialize your code with the cascade you want, and then it does the work for you. Since face detection is such a common case, OpenCV comes with a number of built-in cascades for detecting everything from faces to eyes to hands to legs.

You may use other alternatives to OpenCV, like dlib – that come with Deep Learning based Detection and Recognition models.

### dlib as a code does the following:

1. Use MMOD (Deep Learning) algorithm to find face bounding boxes
2. Find facial landmark points (like eyes, nose, etc.)
3. Use the points to realign the face crops so that it is frontal.
4. Use a Deep Learning model to calculate embeddings from the face crop.
  These embeddings are 128-dimensional vectors. Their nature is such that the same faces will end up closer to each other while different faces will end up far apart.

Also, you may use Dlib face detector in place of OpenCV. Then you can use Pre-trained model like from Facenet, to extract the feature from the face and create embedding for each unique face and assign a name to it. Both Dlib and Facenet score well on accuracy meter.
