# MultiLabel-Image-Classification

### Data:
The given data set consisted a total of 929 colored .png images (and their corresponding labels in the form of .txt files). Each image consisted of one or more type of blood cells namely red blood cell, difficult, gametocyte, trophozoite, ring, schizont and leukocyte.

The distribution of cells across all images is not even. This is a highly skewed data set with red blood cells and trophozoite occurring in almost 65% of images whereas the rest of the five classes occur in only 35% of images.

### Preprocessing:
OpenCV was used to read .png images from the local data folder. Each image was read in Grayscale mode, resized to 150 by 150 pixels, scaled in the range of 0-1 by dividing each pixel value by 255 and then saved in the form of numpy array.
Similarly image labels were also first read and then stored in form of numpy array and then MultiLabelBinarizer() object was used to one-hot encode these target labels in the following order [“red blood cell”, “difficult”, “gametocyte”, “trophozoite”, “ring”, “schizont”, “leukocyte”] so for example, an image containing “red blood cell” and “leukocyte” was encoded as [1,0,0,0,0,0,1].
Finally, both images and their labels were converted to Tensor data type before feeding them into neural network model architecture for swift processing on AWS GPUs.

### Model Architecture:
Using Pytorch’s nn.Module class, layers in my network were structured in the following order: Convolution --- Relu --- Pool --- Convolution --- Relu --- Pool --- Dropout --- Linear --- Linear.
For the first Convolution Layer of the network input channels = 1 was used, since images were read in Grayscale mode, the number of output channels is 10 because we want to apply 10 feature detectors to the images. Also standard kernel size of (3,3) and stride of (1,1) was used.
     
Then Relu activation function was then used to threshold all features to be 0 or greater. This was followed by MaxPool layer that reduced the dimension of image by a factor of 2.
There is a Dropout2d layer in between to regularize the network and hence avoid overfitting during training phase of the network. The final layer of my network is a fully connected standard Linear Layer that computes the score for each of our classes and therefore out_features = 7 in this layer (corresponding to the number of classes that we have in our data set).

#### MultiLabelModel (
<br>(conv1): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
<br>(conv2): Conv2d(10, 20, kernel_size=(3, 3), stride=(1, 1)) (conv2_drop): Dropout2d(p=0.5, inplace=False)
<br>(fc1): Linear(in_features=25920, out_features=1024, bias=True) (fc2): Linear(in_features=1024, out_features=7, bias=True)
) 

### Training:
An instance of the model (defined above) was created with Stochastic Gradient Descent (SGD) as optimizer to update weights and BCEwithLogitsLoss() as criterion for calculating the error. All images were simultaneously passed through the model in one batch to obtain predictions. These predictions as well as the actual target values were then passed through BCEwithLogitsLoss criterion and the results obtained were first passed through Sigmoid function to scale them in a range of 0 to 1 and finally round function was applied to transform the predictions into their one- hot encoded format. A total of 100 epochs were used during training phase and the model with lowest loss (2.8951e-07) was saved in .pt format for portability reasons using TorchScript.
