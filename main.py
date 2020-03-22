from sklearn.preprocessing import StandardScaler

from data_preparation.plot_util import plot_target_with_outlier, plot_regressors_with_outlier, plot_target, \
    plot_regressors, get_autocorr_y, plot_scaled
from db_connection.data_fetch import get_data_from_db
from data_preparation.data_manipulation import plot_preparation, check_for_Null, check_different_ID, \
    prepare_matrix_and_target
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm

#Recupero dati dal database
rows_x, rows_y = get_data_from_db()

#Plotting e valutazioni preliminari sui dati
X_all, y_all, index_primo_plot_x, index_primo_plot_y = plot_preparation(rows_x, rows_y)

#Noto la presenza di pochi outlier dal plot
plot_target_with_outlier(index_primo_plot_y,y_all)
for i in range(0,4):
    plot_regressors_with_outlier(index_primo_plot_x,X_all,i)

#Noto la presenza di None nei dataset
check_for_Null(index_primo_plot_x,X_all,index_primo_plot_y,y_all)

#Noto che i dati non sono allineati
check_different_ID(index_primo_plot_x, index_primo_plot_y)

#Recupero dati genuini e con stesso ID
X, y, index_x, index_y = prepare_matrix_and_target(rows_x, rows_y)
#Plotto i dati senza outlier, senza null ed allineati
plot_target(index_y,y)
for i in range(0,4):
    plot_regressors(index_x,X,i)
print(len(index_y))

#plot autocorrelazione dei target
get_autocorr_y(y)

#ricavo media e varianza dei target
print("Media dei dati: " +str(np.mean(y)))
print("Varianza dei dati: "+ str(np.var(y)))

#matrice di correlazione tra le feature
print("Matrice di correlazione delle feature: ")
print(np.corrcoef(X.transpose()))


#Divisione del dataset in train, validazione e test

#su 9558 dati validi, 8000 sono usati come training, circa 83%
X_train = X[:8001, :]
y_train = y[:8001]

#ID relativi ai dati di training
index_train = index_y[:8001]

#su 9558 dati validi, 500 sono usati come validazione, circa 5%
X_val = X[8001:8501, :]
y_val = y[8001:8501]

#ID relativi ai dati di validazione
index_val = index_y[8001:8501]

#su 9558 dati validi, 500 sono usati come validazione, circa 11%
X_test = X[8501:, :]
y_test = y[8501:]
index_test = index_y[8501:]

#Riscalo feature rimuovendo media e portando a varianza 1
scalerx = StandardScaler()
X_train_scaled = scalerx.fit_transform(X_train)
X_val_scaled = scalerx.transform(X_val)

plot_scaled(index_train,X_train_scaled)

#Riscalo feature rimuovendo media e portando a varianza 1
scalery = StandardScaler()
y_train_scaled = scalery.fit_transform(np.asarray(y_train).reshape(-1,1))

#Creo modello lineare OLS con feature riscalate
modstats_scaled = sm.OLS(y_train_scaled, X_train_scaled)
res_scaled = modstats_scaled.fit()

#Creo modello OLS con feature non riscalate
modstats_not_scaled = sm.OLS(y_train, X_train)
res_not_scaled = modstats_not_scaled.fit()


print("Statistiche modello riscalato")
print(res_scaled.summary())

print("Statistiche modello non riscalato")
print(res_not_scaled.summary())

print("Significatività dei parametri del modello riscalato")
print(res_scaled.pvalues)

print("Significatività dei parametri del modello non riscalato")
print(res_not_scaled.pvalues)


#Uso i modelli sul dataset di validazione

#modello riscalato
y_hat_val_stats_scaled = res_scaled.predict(X_val_scaled)
y_hat_val_stats = scalery.inverse_transform(np.asarray(y_hat_val_stats_scaled).reshape(-1,1))

#modello non riscalato
y_hat_val_stats_not_scaled = res_not_scaled.predict(X_val)

plt.plot(index_val,y_val,'o',label='Inline label')
plt.plot(index_val,y_hat_val_stats,'r',label='Inline label')
plt.plot(index_val,y_hat_val_stats_not_scaled,'b',label='Inline label')
plt.show()


print("RMSE in validazione modello riscalato")
print(rmse(y_hat_val_stats,np.asarray(y_val).reshape(-1,1))[0])

print("RMSE in validazione modello non riscalato")
print(rmse(y_hat_val_stats_not_scaled,y_val))

residual_scaled = y_val - y_hat_val_stats
residual_not_scaled = y_val - y_hat_val_stats_not_scaled

print("Errore medio in validazione modello riscalato")
print(np.mean(residual_scaled))

print("Errore medio in validazione modello non riscalato")
print(np.mean(residual_not_scaled))

#Riscalo il dataset di test
X_test_scaled = scalerx.transform(X_test)

#Uso il modello per fare predizione sul dataset di tast
y_pred = res_scaled.predict(X_test_scaled)
y_pred_rescaled = scalery.inverse_transform(np.asarray(y_pred).reshape(-1,1))

print("RMSE sul dataset di test")
print(rmse(y_pred_rescaled,np.asarray(y_test).reshape(-1,1)))

residual_pred = y_test - y_pred_rescaled

print("Errore medio sul dataset di test")
print(np.mean(residual_pred))

plt.plot(index_test,y_test,'o')
plt.plot(index_test,y_pred_rescaled,'r')
plt.show()