from LinearRegression import LinearRegressor
import matplotlib.pyplot as plt
        
a = LinearRegressor()

# Using a Made Up Linear Data
X_train = [1, 2, 45, 65, 77, 95]
y_train = [1, 2, 45, 65, 77, 95]


w, b= a.train(X_train,y_train)
print(w,b)

y = a.predict([23,45,67,86,32,45])

print(y)

# The correct Y Values
plt.scatter([23,45,67,86,32,45],[23,45,67,86,32,45])

# Plotting our predicted values
plt.plot([23,45,67,86,32,45], y)

plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Line with weight = {w} and bias = {b}")
plt.show()




