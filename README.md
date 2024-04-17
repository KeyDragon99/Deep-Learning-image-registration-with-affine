This Deep Learning project was developed with main goal to create a Neural Network that could potentially return the spatial transformation of an image with respect to another (Moving-Fixed image). This procedure, also known as 
image registration, is a way to spatially match two images in classic computer vision, using classic, widely used algorithms. The purpose of this project was to provide similar or better results to the ones of computer vision,
but produced way faster.
The main dataset used for the project was FIRE (Fundus Image Registration dataset, link: https://projects.ics.forth.gr/cvrl/fire/), which contains high definition image pairs of retinas. 
Together with the images, the dataset provided also hand picked feature points, which were annotated as (x,y) coordinates in txt files for each pair. There were 20 feature points for each image pair, 10 for each image (10 for moving, 10 for fixed).
The pairs were used to generate multiple patch samples which were later used for the training process of the Neural Network.
The Neural Network's (NN) architecture was predefined and passed together with the data that would be used on it for the training.
There are 4 python files, each one containing a different part of the code.
The Network file contains the architecture of the Neural Network.
The DatasetCreator takes as input image files and returns randomly sampled patches (with smaller size) with in each image and saves them in npy form.
The NetworkTrainer file contains the code that takes as input the data and with that, it trains the NN, whose structure is passed through the Network file.
The testing file contains code to receive pre-trained NN and test data and uses the test data on the NN to test its accuracy. It also displays the results by applying the network's predictions on the test samples.
