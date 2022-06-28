# Artificial Neural Network 
Although there are many libraries to create neural networks, I decided to create my own library to better understand how they work.<br>
In this specific case, I used the library to predict whether a woman was suffering from benign or malignant breast cancer.

# Dataset
Training and test data are taken from the [sklearn library](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).<br>
The training and test data contain respectively 512 and 57 samples each consisting of 30 fields plus the outcome (benign or malignant).<br>
Inside the file ```dataset.py``` there are other datasets about other problems.

# Neural Network architecture
The network consists of 30 input neurons, 4 neurons for the first hidden layer, 2 neurons for the second hidden layer and 1 neuron for the output layer.<br>
The bias is not shown in the figure but there is! I did not find any drawing tool that showed it well and I therefore decided to omit it<br>
![ANN structure](/img/ann.png)

# Layer types
The library provides two layer types: 
* fully connected layer
* activation layer <br>
([explanation of layer types](https://theintactone.com/2021/11/28/types-of-layers-convolutional-layers-activation-function-pooling-fully-connected/)).

# Loss function and activation function
I used the mean squared error multiplied by ```1/2``` (The ```1/2``` is included so that exponent is cancelled when we differentiate) as a cost function and as an activation function I used a sigmoid.<br>
mse:<br>
![mse](/img/mse.png)<br>
where ```yi``` is the target value and ```ai``` is the computed value<br>
sigmoid:<br>
![sigmoid](/img/sigmoid.png)<br>

# Backpropagation
These are formulas I used during backpropagation:<br>
![formulas](/img/formulas.png)<br>
Attention, the given formulas are used on the individual elements of the matrices, to use the same formulas on the entire matrix I had to change the order of some multiplications on the code.

# Accuracy
I set the seed of ```numpy.random``` and the ```random_state``` of train_test_split to ensure the reproducibility of the results.<br>
These are the results for:<br>
```net.fit(x_train, y_train, epochs=500, learning_rate=0.1)```<br>
![accuracy](/img/accuracy.png)<br>
almost 93% not bad.

# Improvements
Many improvements can be made, adaptive learning rate, relu, mini batch....
If you have a suggestion that would make this library better, please fork the repo and create a pull request.