import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#=====1) Wczytuję dane, dzielę na dane treningowe i testowe
a = np.loadtxt('Dane\dane5.txt')

x = a[:,[0]]
y = a[:,[1]]


#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)


#=====2) Model liniowy (Model 1), z parametrami a, b 
h = np.hstack([X_train, np.ones(X_train.shape)]) #model y = ax +b 
w = np.linalg.pinv(h) @ y_train # metoda z macierzą odwrotną

#=====3) Błąd oszacowania na zbiorach treningowym, testowym (błąd kwadratowy)
e_train =  1/(2*len(X_train)) * sum(((X_train*w[0] + w[1]) - y_train )**2)
e_test =  1/(2*len(X_test)) * sum(((X_test*w[0] + w[1]) - y_test )**2)

print("Model 1 - błąd treningowy, testowy:", e_train, e_test)



#=====4) Model kwadratowy (Model 2), z parametrami a, b, c
h = np.hstack([X_train*X_train, X_train, np.ones(X_train.shape)]) # model y = ax2 + bx +c 
w1 = np.linalg.pinv(h) @ y_train # metoda z macierza odwrotna

#=====5) Błąd oszacowania na zbiorach treningowym, testowym (Model 2)
e_train = 1/(2*len(X_train)) * sum( ((X_train**2*w1[0])+X_train*w1[1]+w1[2]-y_train)**2)
e_test = 1/(2*len(X_test)) * sum( ((X_test**2*w1[0])+X_test*w1[1]+w1[2]-y_test)**2)

print("Model 2 - błąd treningowy, testowy:", e_train, e_test)



# 5*) Model logarytmiczny (Model 3)
h = np.hstack([1/X_train, np.ones(X_train.shape)]) #model y = (1/x)a + b
w2 = np.linalg.pinv(h) @ y_train

# 5**) Błąd oszacowania na zbiorach treningowym, testowym (Model 3)
e_train = 1/(2*len(X_train)) * sum((1/X_train*w2[0] + w2[1] - y_train)**2)
e_test =  1/(2*len(X_test)) * sum((1/X_test*w2[0] + w2[1] - y_test)**2)

print("Model 3 - błąd treningowy, testowy:", e_train, e_test)



# 5***) Model sinusoidalny (Model 4)
h = np.hstack([np.sin(X_train), np.ones(X_train.shape)]) #model y = a*sin(x)+b
w3 = np.linalg.pinv(h) @ y_train

# 5****) Błąd oszacowania na zbiorach treningowym, testowym (Model 4)
e_train = 1/(2*len(X_train)) * sum((np.sin(X_train*w3[0]) + w3[1] - y_train)**2)
e_test =  1/(2*len(X_test)) * sum((np.sin(X_test)*w3[0] + w3[1] - y_test)**2)

print("Model 4 - błąd treningowy, testowy:", e_train, e_test)



# 5*****) Model cosinusoidalny (Model 5)
h = np.hstack([np.cos(X_train), np.ones(X_train.shape)]) #model y = a*cos(x) + b
w4 = np.linalg.pinv(h) @ y_train

# 5******) Błąd oszacowania na zbiorach treningowym, testowym (Model 5)
e_train = 1/(2*len(X_train)) * sum((np.cos(X_train*w4[0]) + w4[1] - y_train)**2)
e_test =  1/(2*len(X_test)) * sum((np.cos(X_test)*w4[0] + w4[1] - y_test)**2)

print("Model 5 - błąd treningowy, testowy:", e_train, e_test)


#=====6) Porownanie błędów wszystkich Modeli
er1 = 1/(2*len(x))*sum((x*w[0]+w[1]-y)**2 )

er2 = 1/(2*len(x)) * sum( ((x**2*w1[0])+x*w1[1]+w1[2]-y)**2)

er3 = 1/(2*len(x))*sum((1/x*w2[0]+w2[1]-y)**2 )

er4 = 1/(2*len(x))*sum((np.sin(x)*w3[0]+w3[1]-y)**2 )

er5 = 1/(2*len(x))*sum((np.cos(x)*w4[0]+w4[1]-y)**2 )

print(("Błędy modeli 1, 2, 3, 4, 5 na całym zestawie danych: "), er1, er2, er3, er4, er5)

#=====6) Wyrysowanie Modeli na diagramie
plt.plot(x, y, 'ro')
plt.plot(x, w[0]*x + w[1])
plt.plot(x, w1[0]*x*x + w1[1]*x + w1[2])
plt.plot(x, w2[0]*(1/x) + w2[1])
plt.plot(x, w3[0]*np.sin(x) + w3[1])
plt.plot(x, w4[0]*np.cos(x) + w4[1])
plt.show()

