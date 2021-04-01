# Creating Artificial Neural Network from scratch
* Project: To classify hand written images from MNIST database using artificial neural network from scrach
* Data source: [MNIST Dataset to download](http://yann.lecun.com/exdb/mnist/).  The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. This contains grayscale samples of handwritten digits of size 28 x 28 and has a training set of 60,000 examples, and a test set of 10,000 examples. 
* Here is the sample gray scale image with respective labels 
![](1mages/sample_digits.png)
* Below is the ordered list of code components developed to build the model from scratch:
1. ReLU (Rectified Linear Unit) activation funtion
2. Gradient of ReLU Activation
3. Linear Activation
4. Gradient of Linear Activation
5. Softmax-Cross Entropy Loss
6. Derivative of Softmax-Cross Entropy Loss
7. Dropout Forward
8. Dropout Backward
9. Batch Norm Forward
10. Batch Norm Backward
11. Initialization of Parameter 
12. Adam Optimizer: Initialization and Update
13. Forward Propagation
14. Back Propagation
15. Update Parameters using Batch Gradient
16. Putting together (Forward + Back Propagation)
17. Class Prediction
18. Training and Testing
19. Accuracy
