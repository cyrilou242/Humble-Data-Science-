import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=4, suppress=False)

"""
Exercices Fitting
Cyril de Catheu
Last update 06/03/2018
"""


"""
#############################################################################
############################## MOINDRES CARRES #############################
#############################################################################
"""

def moindrescarres(x,y):
    """Moindres carrés classique de 2 vecteurs de la même taille"""
    X = np.transpose(np.vstack((np.array(x),np.array(np.ones(x.size)))))
    inv = np.linalg.inv(np.dot(np.transpose(X),X))
    xty = np.dot(np.transpose(X), y)
    h = np.dot(inv,xty)
    return h

# on défini un loader de jeu de données
def loader(filepath):
    """Chargement d'un jeu de données, assumé par la suite de dimension 2"""
    data = np.loadtxt(filepath)
    return data

def zplot(f, a, b, n, color="r"):
    """Tracé d'une fonction sur un intervalle donné avec un nb de points données"""
    """Pas forcément utile pour tracer des droites mais classique"""
    x = []
    y = []
    abscisse = a
    pas  = (b - a)/ n
    for k in range(0, n + 1):
        x.append(abscisse)
        y.append(f(abscisse))
        abscisse += pas
    plt.plot(x, y, color)


def trace(filepath,ransac_run = False, delta=0.2, itermax = 10):
    """Fonction principale de tracé. On peut tracer le résultat du ransac en passant l'attribut à True"""
    #tracé des points
    data = loader(filepath)
    X = data[:,0]
    Y =  data[:,1]
    plt.plot(X,Y,'b.')

    #tracé de la droite
    h = moindrescarres(X,Y)
    def f(x):
         y = x * h[0] + h[1]
         return y
    zplot(f,0,100,100)

    if ransac_run:
        h2 = np.array([0.0,0.0])
        h2[0],h2[1],C = ransac(X,Y, delta, itermax)
        print(h2)
        def f2(x):
            y = x * h2[0] + h2[1]
            return y
        zplot(f2,0,100,100,color="g")

    return plt.show()

#trace("data.txt")
#trace("dataOutliers.txt")


"""
#############################################################################
############################## RANSAC #############################
#############################################################################
"""

def droite(s1,t1,s2,t2):
    """Fonction qui renvoie les coefficients de la droite définie par 2 points"""
    X = np.array([s1,s2])
    Y = np.array([t1,t2])
    #on reutilise moindres carres pour 2 points
    return moindrescarres(X,Y)

def ransac(x,y,delta, itermax):
    """Algorithme Ransac"""
    long = x.size

    Cmax =[]
    hmax =np.array([0.0,0.0])
    #boucle principale à initiaiser en fonction de la longueur de C, de
    for iter in range(itermax):
        #on choisi 2 points au hasard
        idx = np.random.randint(long, size=2)
        print(idx)
        h = droite(x[idx[0]],y[idx[0]],x[idx[1]],y[idx[1]])
        # on défini la fonction qui dépend de h
        def f(x):
            y = x * h[0] + h[1]
            return y
        #on calcule les distances pour tous les points
        Ctemp = []
        for i in range(long):
            if abs(f(x[i])-y[i]) < delta:
                Ctemp.append((x[i],y[i]))

        if not Cmax:
            Cmax = [e for e in Ctemp]
            hmax[0] = h[0]
            hmax[1] = h[1]
            print("entrée dans cas Cmax vide")
        elif not Ctemp:
            #do nothing, no point were selected
            print("entrée dans cas Ctemp vide")
        elif len(Ctemp) > len(Cmax) :
            Cmax = [e for e in Ctemp]
            hmax[0] = h[0]
            hmax[1] = h[1]
    print(hmax)
    return(hmax[0],hmax[1],Cmax)


"""
#############################################################################
############################## TESTS #############################
#############################################################################
"""

trace("data.txt")
trace("dataOutliers.txt")
trace("dataOutliers.txt", ransac_run=True, delta=0.2, itermax=7)
trace("dataOutliers.txt", ransac_run=True, delta=0.2, itermax=2)
trace("dataOutliers.txt", ransac_run=True, delta=300, itermax=7)
