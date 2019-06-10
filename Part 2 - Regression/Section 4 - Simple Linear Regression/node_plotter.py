import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X_train = [[float(x)] for x in sys.argv[1].split(',')]
print(type(X_train))
y_train = [float(x) for x in sys.argv[2].split(',')]
y_predict = [float(x) for x in sys.argv[3].split(',')]

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_predict, color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
