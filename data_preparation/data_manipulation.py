import numpy as np

from data_preparation.plot_util import reg_selector


def check_different_ID(index_x, index_y):

    print("Il numero di misure è: "+str(len(index_x)))
    print("L'ultimo indice dei target è: "+str(index_y[len(index_y)-1]))
    print("L'ultimo indice dei regressori è: " + str(index_y[len(index_y) - 1]))
    print("Gli indici mancanti dei target sono: 200, 201, 202")
    print("Gli indici mancanti dei target sono: 100, 101, 102" )

def check_for_Null(index_x,X,index_y,y):

    for it in range(0,len(index_y)):

        if y[it] is None:
            print("Il target con ID "+str(index_y[it])+" ha valore None")
        if X[it,0] is None:
            print("La misura di "+reg_selector(0) +" con ID "+str(index_x[it])+" ha valore None")
        if X[it,1] is None:
            print("La misura di "+reg_selector(1) +" con ID "+str(index_x[it])+" ha valore None")
        if X[it,2] is None:
            print("La misura di "+reg_selector(2) +" con ID "+str(index_x[it])+" ha valore None")
        if X[it,3] is None:
            print("La misura di "+reg_selector(3) +" con ID "+str(index_x[it])+" ha valore None")




def plot_preparation(rows_x, rows_y):
    y = []

    AT = []
    V = []
    AP = []
    RH = []

    X = []

    index_x = []
    index_y = []

    # schema regressori:
    # posizione 0 : indice
    # posizione 1 : AT
    # posizione 2 : V
    # posizione 3 : AP
    # posizione 4 : RH

    # schema target:
    # posizione 0 : indice
    # posizione 1 : PE

    for elemx, elemy in zip(rows_x, rows_y):
        index_y.append(elemy[0])
        y.append(elemy[1])

        index_x.append(elemx[0])
        AT.append(elemx[1])
        V.append(elemx[2])
        AP.append(elemx[3])
        RH.append(elemx[4])

        X = np.array([AT, V, AP, RH])
        X = X.transpose()

    return X, y, index_x, index_y


def prepare_matrix_and_target(rows_x, rows_y):
    # lista target
    y = []

    # liste temporanee per i regressori
    AT = []
    V = []
    AP = []
    RH = []

    # matrice finale dei regressori
    X = []

    # indici delle misure di regressori e target
    index_x = []
    index_y = []


    # schema regressori:
    # posizione 0 : indice
    # posizione 1 : AT
    # posizione 2 : V
    # posizione 3 : AP
    # posizione 4 : RH

    # schema target:
    # posizione 0 : indice
    # posizione 1 : PE

    #iteratori di variabili e target
    itx = 0
    ity = 0
    while itx<len(rows_y) and ity < len(rows_y):

        #singola tupla di dati
        elemx = rows_x[itx]
        elemy = rows_y[ity]


        #uno dei due indici ha delle misure mancanti
        if elemx[0] != itx and elemx[0]!=elemy[0]:

            ity = elemx[0]
            elemy = rows_y[ity]

        if elemy[0]!= ity and elemx[0] != elemy[0]:

            itx = ity
            elemx = rows_x[itx]




        if elemx[1] is not None \
                and elemx[2] is not None \
                and elemx[3] is not None \
                and elemx[4] is not None \
                and elemy[1] is not None:

            # ignoro outlier di AP

            if elemx[3] > 1200:
                itx += 1
                ity += 1
                continue
            # ignoro outlier target
            if elemy[1] > 700:
                itx += 1
                ity += 1
                continue

            index_y.append(elemy[0])
            y.append(elemy[1])

            index_x.append(elemx[0])
            AT.append(elemx[1])
            V.append(elemx[2])
            AP.append(elemx[3])
            RH.append(elemx[4])

            X = np.array([AT, V, AP, RH])
            X = X.transpose()
        itx += 1
        ity += 1


    return X, y, index_x, index_y
