#A class for Loss functions
class LossFunction:

    #Init function. Takes in number . Takes in num of rows
    def __init__(self, method='MSE', m=2):
        self.m = m #m = number of rows || number of datapoints

    # Mean Squared Error. One of the loss functions
    def Mean_Squared_Error(self, pred, actual): 
        
        #Takes in self. list of Predicted values (ŷ). List of Actual values (y) 

        loss = 0 # MSE = 1/(2m)* Σ((ŷ-y)^2), iterated m times

        for i in range(self.m):
            loss = loss+(actual[i]-pred[i])**2

        FinalLoss = loss/(self.m)
        return FinalLoss


        

def Gradient_Descent(x, y, alpha=0.0001, epochs=300):

    m = len(x)

    weights = 0
    bias= 0

    for _ in range(epochs):

        updated_weights = updated_bias = 0

        for i in range(m):
        
            updated_weights += -(2/m)*x[i]*(y[i]-(weights*x[i]+bias))
            updated_bias += -(2/m)*(y[i]-(weights*x[i]+bias))

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

    def train(self, x, y):

        self.weights, self.bias = Gradient_Descent(x, y, self.alpha)
        return self.weights,self.bias

    def gettingValues(self, x):
        y = self.weights*x + self.bias
        return y

    def predict(self, x):

        if self.weights == None:
            raise  NotImplementedError("Model not Trained. Use model.train() to train it")
        
        PREDICTED_Value = []

        for i in x:
            print(i)
            PREDICTED_Value.append(self.gettingValues(i))

        return PREDICTED_Value

