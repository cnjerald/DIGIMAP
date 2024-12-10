The dataset used is the Intel Image Classification Dataset from Kaggle. The Images in the dataset were designed for multi-class image classification involving six categories: buildings, forest, glacier, mountain, sea, and street. The dataset is then organized into training, testing, and validation subsets, with each subset containing images for the respective categories. This structure allows efficient training testing, and validation of machine learning models. The images are kept in directories according to their labels, which makes it easily compatible with libraries like TensorFlow and PyTorch. The dataset is appropriate for improving methods for image classification or testing algorithms.

Prior to training, data augmentation was performed in the training set, preventing overfitting during model training. Augmentation is achieved by introducing variability in the training data so that the model is resilient against variations in real-world scenarios. These augmentations include:
1. Adding a random black patch
2. Rotating the image
3. Shifting the image
4. Flipping the image

Before getting started, make sure you have installed the following using pip install [library name]:
1. Tensorflow is a library used for machine learning. Install with pip install tensorflow.
2. Matplotlib is a plotting library used for visualizations such as graphs. Install with pip install matplotlib
3. NumPy is a fundamental library for numerical computing in python. Install with pip install numpy
4. CV2 from OpenCV is used for loading, manipulating, and displaying images. Install with pip install opencv-python

Files provided:
1. seg_test and seg_train: These folders contain the train and test image sets used for the convolutional neural network (CNN)
2. image_class.ipynb: Training the CNN takes a substantial amount of time (up to 5 hours in our tests) so we have provided this notebook containing all the results from our runs
3. image_class.py: Main deliverable
