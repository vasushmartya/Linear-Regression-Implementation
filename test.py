import model

X = [1,2,3,4]
y = [2,4,7,8]
X_test = [7, 8, 9, 10]
y_test = [14, 17, 19, 20]


regressor = model.SimpleLinearRegression()

regressor.fit(X, y)

regressor.coefficients()

predictions = regressor.predict(X_test)

r2 = regressor.r_squared(y_test, predictions)
print(r2)