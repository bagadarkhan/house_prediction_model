import matplotlib.pyplot
import pandas
import numpy 

data=pandas.read_csv("ev_verileri.csv")

print(data.head())

x=data[["Alan"]]
y=data[["Fiyat"]] 



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x_train,y_train)

tahmin=model.predict(x_test)
print(tahmin)


matplotlib.pyplot.scatter(x_test,y_test,color='blue',label="Gerçek değerler")
matplotlib.pyplot.plot(x_test,tahmin,color="Green",label="Tahminler")
matplotlib.pyplot.xlabel("Alan")
matplotlib.pyplot.ylabel("fiyat")

metrekare=float(input("Evin Metrekaresini giriniz: "))
tahmin_fiyat=model.predict(([[metrekare]]))

print(f"{metrekare} m2 evin tahmini fiyatı: {tahmin_fiyat[0][0]:,.2f}TL")







