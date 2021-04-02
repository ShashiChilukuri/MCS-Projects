# Creating Artificial Neural Network from scratch
* Project: To classify hand written images from MNIST database using artificial neural network from scrach
* Data source: [MNIST Dataset to download](http://yann.lecun.com/exdb/mnist/).  The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. This contains grayscale samples of handwritten digits of size 28 x 28 and has a training set of 60,000 examples, and a test set of 10,000 examples. Here is the sample gray scale image with respective labels 
![](1mages/sample_digits.png)
* Below is the ordered list of code components developed to build the model from scratch:
1. Load dataset
2. ReLU (Rectified Linear Unit) activation funtion
3. Gradient of ReLU Activation
4. Linear Activation
5. Gradient of Linear Activation
6. Softmax-Cross Entropy Loss
7. Derivative of Softmax-Cross Entropy Loss
8. Dropout Forward
9. Dropout Backward
10. Batch Norm Forward
11. Batch Norm Backward
12. Initialization of Parameter 
13. Adam Optimizer: Initialization and Update
14. Forward Propagation
15. Back Propagation
16. Update Parameters using Batch Gradient
17. Putting together (Forward + Back Propagation)
18. Class Prediction
19. Training and Testing
20. Accuracy
