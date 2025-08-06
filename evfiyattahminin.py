import matplotlib.pyplot as plt
import pandas
import numpy
import statsmodels.api as sm
#okuduk
df=pandas.read_csv("ev_verileri.csv")
#işlem yapabilmek için parçaladık
X=df[["Alan","OdaSayısı","BinaYaşı"]]### burada kat değişkeninin önemsiz oldugunu kesfettik
Y=df[["Fiyat"]]

#istatistiksel analiz aşaması hangi değişkenler anlamlı???
X=sm.add_constant(X)

model=sm.OLS(Y,X).fit()

print(model.summary())

from sklearn.model_selection import train_test_split
#modeli eğittik
X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
#modeli inşaa ettik
model=LinearRegression()
model.fit(X_train, Y_train)
#tahminler yapar hale geldi
Y_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

#hata payları
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)#ortalama hata payı
r2=r2_score(Y_test,Y_pred)#uyumu ölçer

#grafiği çizdik
plt.figure(figsize=(8,6))
plt.scatter(Y_test, Y_pred, color="blue", label="Tahminler")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', label="Mükemmel Tahmin")
plt.xlabel("Gerçek Fiyatlar (₺)")
plt.ylabel("Tahmin Edilen Fiyatlar (₺)")
plt.title("Gerçek ve Tahmin Fiyat Karşılaştırması")
plt.legend()
plt.show()

#Modeli Yorumlama#
print("Katsayılar:",model.coef_)
print("Sabit katsayılar:",model.intercept_)






