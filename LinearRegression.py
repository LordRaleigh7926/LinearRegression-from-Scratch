#A class for Loss functions
class LossFunction:

    # Mean Squared Error. One of the loss functions
    def Mean_Squared_Error(self, pred, actual): 
        
        #Takes in self. list of Predicted values (ŷ). List of Actual values (y) 

        # m stores number of DataPoints
        m = len(pred) 

        # MSE = 1/(2m)* Σ((ŷ-y)^2), iterated m times. For more info search MSE formula
        loss = 0 

        # Iterating m times
        for i in range(m): 
            loss = loss+(actual[i]-pred[i])**2

        FinalLoss = loss/(m)

        return FinalLoss


''' 
Gradient Descent Function.
Used to minimize error and get to the best weights and bias.
Takes in x, y and the learning rate and the number of epochs
Formula for Gradient Descent for Linear Regression
'''
def Gradient_Descent(x, y, alpha=0.0001, epochs=300):

    # m stores number of DataPoints
    m = len(x)

    #Setting the main Bias and Weight to 0
    weights = 0
    bias= 0

    # For loop for epochs
    for _ in range(epochs):

        # Setting the updated weight and updated bias to 0 after every iteration
        updated_weights = updated_bias = 0

        # Loop for adding the derivated error for all the datapoints
        for i in range(m):
        
            updated_weights += -(2/m)*x[i]*(y[i]-(weights*x[i]+bias))
            updated_bias += -(2/m)*(y[i]-(weights*x[i]+bias))

        #Updating the weights and biases
        weights = weights - updated_weights*alpha
        bias = bias - updated_bias*alpha

    
    return weights, bias


# The Linear Regressor
class LinearRegressor:

    # Takes in the loss function and the learning rate
    def __init__(self, loss_function='Mean_Squared_Error', learning_rate=0.0001):
        
        self.alpha = learning_rate
        self.J = loss_function
        self.bias = self.weights = None

    # Used to train the model and set the weights and bias
    # It uses the function Gradient_Descent
    def train(self, x, y):

        self.weights, self.bias = Gradient_Descent(x, y, self.alpha)
        return self.weights,self.bias

    # Used to predict a single y value for x
    # Used in predict function
    def gettingValues(self, x):
        y = self.weights*x + self.bias
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

