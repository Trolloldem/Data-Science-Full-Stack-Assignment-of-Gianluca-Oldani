import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

def plot_scaled(index_train,X_train_scaled):

    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
    fig.tight_layout(pad=3.0)
    axs[0, 0].plot(index_train, X_train_scaled[:, 0], 'b')
    axs[0, 0].set_title('AT Standardizzata')
    axs[0, 0].set(xlabel='sample_ID', ylabel='AT')
    axs[0, 1].plot(index_train, X_train_scaled[:, 1], 'b')
    axs[0, 1].set_title('V Standardizzata')
    axs[0, 1].set(xlabel='sample_ID', ylabel='V')
    axs[1, 0].plot(index_train, X_train_scaled[:, 2], 'b')
    axs[1, 0].set_title('AP Standardizzata')
    axs[1, 0].set(xlabel='sample_ID', ylabel='AP')
    axs[1, 1].plot(index_train, X_train_scaled[:, 3], 'b')
    axs[1, 1].set_title('RH Standardizzata')
    axs[0, 1].set(xlabel='sample_ID', ylabel='RH')
    plt.show()

def get_autocorr_y(y):
    plot_acf(np.asarray(y))
    plt.suptitle("Autocorrelazione dei target")
    plt.show()

def reg_selector(col):
    if col == 0:
        return 'AT'
    if col == 1:
        return  'V'
    if col == 2:
        return 'AP'
    if col == 3:
        return 'RH'
def plot_target_with_outlier(index_y,y):

    #plot con outlier
    plt.plot(index_y, y, 'o')
    plt.suptitle('Misure di PE con outlier')
    plt.ylabel('PE')
    plt.xlabel('sample_ID')
    plt.show()

    #plot con dati genuini
    plt.plot(index_y, y,'o')
    plt.suptitle('Misure di PE senza outlier')
    plt.ylabel('Misure PE')
    plt.xlabel('sample_ID')
    plt.axis([0,len(index_y),410,510])
    plt.show()

def plot_regressors_with_outlier(index_x,X, col):

    #plot con outlier
    plt.plot(index_x, X[:,col], 'o')

    if col!=2:
        plt.suptitle('Misure variabile '+reg_selector(col))
    else:
        plt.suptitle('Misure variabile ' + reg_selector(col)+' con outlier')

    plt.ylabel("Misure "+reg_selector(col))
    plt.xlabel('sample_ID')
    plt.show()

    #plot dato genuino per AP
    if col == 2:
        plt.plot(index_x, X[:, col], 'o')
        plt.suptitle('Misure variabile AP senza outlier')
        plt.ylabel('Misure AP')
        plt.xlabel('sample_ID')
        plt.axis([0, len(index_x), 975, 1050])
        plt.show()

def plot_target(index_y,y):

    #plot
    plt.plot(index_y, y, 'o')
    plt.suptitle('Misure di PE')
    plt.ylabel('PE')
    plt.xlabel('sample_ID')
    plt.show()


def plot_regressors(index_x, X, col):

    # plot con outlier
    plt.plot(index_x, X[:, col], 'o')
    plt.suptitle('Misure variabile ' + reg_selector(col))
    plt.ylabel("Misure " + reg_selector(col))
    plt.xlabel('sample_ID')
    plt.show()



