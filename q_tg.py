import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv("etmgeg_310.txt", sep=",")
dataf = df.dropna()

# print(dataf)
# dataf.info()
# print(dataf.describe())
# print(dataf.isnull().sum())

# cr = dataf.corr()
# cr.to_excel("correlation_matrix_310.xlsx", sheet_name='Sheet_name_1') 

Y = dataf["TG"]
X = dataf["Q"].values.reshape(Y.size, 1)

plt.scatter(X, Y, 12)
plt.title("Figuur 1")
plt.xlabel("Globale straling in J/cm2")
plt.ylabel("Etmaalgemiddelde temperatuur in deg C")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
lr = LinearRegression()
lr.fit(X_train,Y_train)

plt.scatter(X_train, Y_train, 12, c="orange")
plt.title("Linear regression train Q-TG")
plt.plot(X_train, lr.predict(X_train))
plt.xlabel("Globale straling in J/cm2")
plt.ylabel("Etmaalgemiddelde temperatuur in deg C")
plt.show()

plt.scatter(X_test, Y_test,color='r')
plt.title("Linear regression test Q-TG")
plt.plot(X_test, lr.predict(X_test))
plt.xlabel("Globale straling in J/cm2")
plt.ylabel("Etmaalgemiddelde temperatuur in deg C")
plt.show()

print("Score train data: " + str(round(lr.score(X_train, Y_train),2)))
print("Score test data: " + str(round(lr.score(X_test, Y_test),2)))

print(lr.predict([[4000]]))

pickle.dump(lr, open('model_Q-TG.pkl','wb'))