import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("etmgeg_310.txt", sep=",")
dataf = df.dropna()

Y = dataf["UG"]
X = dataf["TG"].values.reshape((Y.size, 1))

plt.scatter(X, Y)
plt.title("Figuur 2")
plt.xlabel("Etmaalgemiddelde temperatuur in 0.1 deg C")
plt.ylabel("Etmaalgemiddelde relatieve vochtigheid in %")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
lr = LinearRegression()
lr.fit(X_train,Y_train)

plt.scatter(X_train, Y_train, c="orange")
plt.title("Linear regression train TG-UG")
plt.plot(X_train, lr.predict(X_train))
plt.xlabel("Etmaalgemiddelde temperatuur in 0.1 deg C")
plt.ylabel("Etmaalgemiddelde relatieve vochtigheid in %")
plt.show()

plt.scatter(X_test, Y_test, color='r')
plt.title("Linear regression test TG-UG")
plt.plot(X_test, lr.predict(X_test))
plt.xlabel("Etmaalgemiddelde temperatuur in 0.1 deg C")
plt.ylabel("Etmaalgemiddelde relatieve vochtigheid in %")
plt.show()

print("Score train data: " + str(round(lr.score(X_train, Y_train),2)))
print("Score test data: " + str(round(lr.score(X_test, Y_test),2)))

print(lr.predict([[320]]))

pickle.dump(lr, open('model_TG-UG.pkl','wb'))