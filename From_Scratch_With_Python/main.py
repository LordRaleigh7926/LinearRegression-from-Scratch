from LinearRegression import LinearRegressor, LossFunction

model = LinearRegressor(epochs=900)
Lossfunc = LossFunction()

# Using a Made Up Linear Data
X_train = [[1,1,1,1],[4,4,4,4]]
y_train = [1,4]
X_test = [[7,7,7,7]]
y_test = [7]


model.train(X_train,y_train)

print(model.weights, model.bias)

y = model.predict(X_test)

loss = Lossfunc.Mean_Squared_Error([7], y)

print(loss)
print(y)





