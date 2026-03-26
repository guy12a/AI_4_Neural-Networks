import nn

class PerceptronModel(object):
    def __init__(self, dimension):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimension` is the dimensionality of the data.
        For example, dimension=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimension)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, point):
        """
        Calculates the score assigned by the perceptron to a data point `point`.

        Inputs:
            point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(),point)

    def get_prediction(self, point):
        """
        Calculates the predicted class for a single data point `point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(point))  >= 0:
            return 1
        else:
            return -1

    def train_model(self, data):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        finished = False
        while not finished:
            allCorrect = True
            #Get one data entry at a time, and for each get the prediction.
            #If prediction is wrong, have to go again, and also update the weights
            for x, y in data.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    allCorrect = False
                    self.get_weights().update(nn.as_scalar(y),x)
            if allCorrect:
                finished = True
                


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hiddenLayerSize = 70

        #first layer, (1) - > (hiddenLayerSize)
        self.W1 = nn.Parameter(1,hiddenLayerSize)
        self.b1 = nn.Parameter(1,hiddenLayerSize)

        #second layer, (hiddenLayerSize) - > (hiddenLayerSize)
        self.W2 = nn.Parameter(hiddenLayerSize,hiddenLayerSize)
        self.b2 = nn.Parameter(1,hiddenLayerSize)

        #output, (hiddenLayerSize) - > (1)
        self.W3 = nn.Parameter(hiddenLayerSize,1)
        self.b3 = nn.Parameter(1,1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # returning: relu( relu( x*W1 + b1 )*W2 + b2 )*W3 + b3
        first = nn.ReLU(nn.AddBias(self.b1,nn.Linear(x,self.W1)))
        second = nn.ReLU(nn.AddBias(self.b2,nn.Linear(first,self.W2)))
        return nn.AddBias(self.b3,nn.Linear(second,self.W3))


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y)

    def train_model(self, data):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #sets learnRate as negative as i try to minimize the loss function
        learnRate = -0.007
        batch = 20

        finished = False
        while not finished:
            for x, y in data.iterate_once(batch):
                loss = self.get_loss(x,y)
                #gets gradient per each variable
                gW1, gb1, gW2, gb2, gW3, gb3 = nn.gradients([self.W1, self.b1,self.W2, self.b2,  self.W3, self.b3],loss)
                #updates each in the opposite direction to the gradient
                self.W1.update(learnRate,gW1)
                self.W2.update(learnRate,gW2) 
                self.W3.update(learnRate,gW3)
                self.b1.update(learnRate,gb1)
                self.b2.update(learnRate,gb2)
                self.b3.update(learnRate,gb3)
            #once loss for all model is below 0.019 (for safety), stops
            for x,y in data.iterate_once(200):
                modelLoss = nn.as_scalar(self.get_loss(x,y))
            if modelLoss <0.019:
                finished = True
            


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        
        hiddenLayerSize = 100

        #first layer, (784) - > (hiddenLayerSize)
        self.W1 = nn.Parameter(784,hiddenLayerSize)
        self.b1 = nn.Parameter(1,hiddenLayerSize)

        #second layer, (hiddenLayerSize) - > (hiddenLayerSize)
        self.W2 = nn.Parameter(hiddenLayerSize,hiddenLayerSize)
        self.b2 = nn.Parameter(1,hiddenLayerSize)

        #third layer, (hiddenLayerSize) - > (hiddenLayerSize)
        self.W4 = nn.Parameter(hiddenLayerSize,hiddenLayerSize)
        self.b4 = nn.Parameter(1,hiddenLayerSize)

        #output, (hiddenLayerSize) - > (10)
        self.W3 = nn.Parameter(hiddenLayerSize,10)
        self.b3 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #3 hidden layers of ReLU
        first = nn.ReLU(nn.AddBias(self.b1,nn.Linear(x,self.W1)))
        second = nn.ReLU(nn.AddBias(self.b2,nn.Linear(first,self.W2)))
        third = nn.ReLU(nn.AddBias(self.b4,nn.Linear(second,self.W4)))
        return nn.AddBias(self.b3,nn.Linear(third,self.W3))

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x),y)

    def train_model(self, data):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #sets learnRate as negative as i try to minimize the loss function
        learnRate = -0.0065
        batch = 50

        finished = False
        while not finished:
            for x, y in data.iterate_once(batch):
                loss = self.get_loss(x,y)
                #gets gradient per each variable
                gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4 = nn.gradients([self.W1, self.b1 ,self.W2, self.b2, self.W3, self.b3, self.W4, self.b4],loss)
                
                #updates each in the opposite direction to the gradient
                self.W1.update(learnRate,gW1)
                self.W2.update(learnRate,gW2)  
                self.W3.update(learnRate,gW3)
                self.W4.update(learnRate,gW4)
                self.b1.update(learnRate,gb1)
                self.b2.update(learnRate,gb2)
                self.b3.update(learnRate,gb3)
                self.b4.update(learnRate,gb4)

            #once accuracy is above 0.975, stops
            if data.get_validation_accuracy() > 0.975:
                finished = True

