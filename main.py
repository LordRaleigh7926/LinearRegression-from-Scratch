import numpy as np 
from functions import LinearRegressor
import matplotlib.pyplot as plt
        
a = LinearRegressor()
X_train = [1, 2, 45, 65, 77, 95]
y_train = [1, 2, 45, 65, 77, 95]

plt.scatter(X_train,y_train)

w, b= a.train(X_train,y_train)
print(w,b)

y = a.predict([23,45,67,86,32,45])

print(y)

# Create the plot
plt.plot([23,45,67,86,32,45], np.reshape(y,(6,)))

# Add labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Line with weight = {w} and bias = {b}")

# Show the plot
plt.show()




