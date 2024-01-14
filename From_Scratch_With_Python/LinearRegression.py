#A class for Loss functions
class LossFunction:

    # Mean Squared Error. One of the loss functions
    #Takes in self. list of Predicted values (ŷ). List of Actual values (y) 
    def Mean_Squared_Error(self, actual, pred): 
        

        # m stores number of DataPoints
        m = len(pred) 

        # MSE = 1/(m)* Σ((y-ŷ)^2), iterated m times. For more info search MSE formula
        loss = 0 

        # Iterating m times
        for i in range(m): 
            # Adding the squared errors
            loss = loss+(actual[i]-pred[i])**2

        FinalLoss = loss/(m)

        return FinalLoss


''' 
Gradient Descent Function.
Used to minimize error and get to the best weights and bias.
Takes in x, y and the learning rate and the number of epochs
Formula for Gradient Descent for Linear Regression
'''
def Gradient_Descent(x, y, alpha=0.0001, epochs=1000):

    # m stores number of DataPoints
    m = len(x)

    # n stores number of features
    n = len(x[0])

    #Setting the main Bias and Weight to 0
    bias= 0

    # weights is a list where there are n number of 0s 
    weights = []
    for i in range(n):
        weights.append(0)

    # For loop for epochs
    for _ in range(epochs):

        # Calculating the error
        for i in range(m):

            prediction = bias

            for k in range(n):
                prediction += weights[k] * x[i][k]
            
            error = y[i] - prediction


        # Setting the updated weight and updated bias to 0 after every iteration
        updated_bias = 0
        updated_weights = []
        for i in range(n):
            updated_weights.append(0)

        # Loop for adding the derivated error for all the datapoints
        for i in range(m):
            
            # Calculating the Derivatives
            for k in range(len(weights)):
                updated_weights[k] += -(2/m)*x[i][k]*error
            updated_bias += -(2/m) * error

        # Updating the weights and biases
        for i in range(len(x[0])):
            weights[i] = weights[i] - updated_weights[i]*alpha
        bias = bias - updated_bias*alpha

    
    return weights, bias


# The Linear Regressor
class LinearRegressor:

    # Takes in the learning rate and number of epochs
    # Default epochs = 1000
    # Default learning_rate = 0.0001
    def __init__(self, learning_rate=0.0001, epochs=1000):
        
        self.epochs = epochs
        self.alpha = learning_rate
        self.bias = self.weights = None

    # Used to train the model and set the weights and bias
    # It uses the function Gradient_Descent
    def train(self, x, y):

        self.weights, self.bias = Gradient_Descent(x, y, self.alpha, self.epochs)

    # Used to predict a single y value for x
    # Used in predict function
    def gettingValues(self, x):

        # Initializing y as bias
        y = self.bias

        # Adding mᵢ*xᵢ till i=number of features
        for i in range(len(x)):
            y += self.weights[i]*x[i]
        
        # Returning the predicted output for the datapoint
        return y 

    # This is of course used to predict the set of x values
    # As mentioned earlier it uses the function gettingValues 
    def predict(self, x):

        # if the model is not trained then it will throw an NotImplemented error
        if self.weights == None:
            raise  NotImplementedError("Model not Trained. Use model.train() to train it")
        
        PREDICTED_Value = []

        # Loop for getting all the predicted value individually
        for i in x:
            PREDICTED_Value.append(self.gettingValues(i))

        return PREDICTED_Value

