"""
Exercice ACP
Cyril de Catheu
Last update 14/02/2018
"""


import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import math
np.set_printoptions(precision=4,suppress=False)

"""
Sujet : 
- Matrice centrée réduite
- Matrice des corrélations
- Valeurs propres et vecteurs propres
- Calculs des coordonnées projetées sur (C1,C2) et représentation graphique
- Calcul du % d'inertie totale exprimée par (C1,C2)
- Calcul de la matrice des corrélations des données par rapport à (C1,C2)
- Cercle des corrélations
"""

#Initialisation des infos.
varNames = np.array(['calcium', 'magnesium','sodium','potassium','sulfate','no3','HCo3','chlore'])
m = np.matrix([[78, 24,5,1,10,3.8,357 ,4.5],
[48,11,34 ,1,16 ,4 ,183,50],
[71,5.5,11.2,3.2,5,1,250,20],
[89,31,17,2,47,0,360,28],
[4.1,1.7,2.7,0.9,1.1,0.8,25.8,0.9],
[85,80,385,65,25,1.9,1350,285],
[26.5,1,0.8,0.2,8.2,1.8,78.1,2.3],
[9.9,6.1,9.4,5.7,6.9,6.3,65.3,8.4],
[63,10.2,1.4,0.4,51.3,2,173.2,1],
[234,70,43,9,635,1,292,62],
[170,92,650,130,31,0,2195,387],
[63,10.2,1.4,0.4,51.3,2,173.2,10],
[46.1,4.3,6.3,3.5,9,0,163.5,3.5],
[108,14,3,1,13,12,350,9],
[84,23,2,1,27,0.2,341,3],
[486,84,9.1,3.2,1187,2.7,403,8.6],
[86,3,17,1,7,19,256,21],
[125,30.1,126,19.4,365,0,164.7,156],
[241,95,255,49.7,143,1,1685.4,38],
[253,11,7,3,25,1,820,4],
[48.1,9.2,12.6,0.4,9.6,0,173.3,21.3],
[54.1,31.5,8.2,0.8,15,6.2,267.5,13.5],
[110.8,9.9,8.4,0.7,39.7,35.6,308.8,8],
[25.7,10.7,8,0.4,9.6,3.1,117.2,12.4],
[12.3,2.6,2.5,0.6,10.1,2.5,41.6,0.9],
[46,28,6.8,1,5.8,6.6,287,2.4],
[208,55.9,43.6,2.7,549.2,0.45,219.6,74.3],
[19.8,1.8,1.7,1.8,14.2,1.5,56.5,0.3],
[36,13,2,0.6,18,3.6,154,2.1],
[32.5,6.1,4.9,0.7,1.6,4.3,135.5,1],
[354,83,653,22,1055,0,225,982],
[46.1,4.3,6.3,3.5,9,0,163.5,3.5],
[36,19,36,6,43,0,195,38],
[8,10,33,4,20,0.5,84,37],
[46,33,430,18.5,10,8,1373,39],
[5.2,2.43,14.05,1.15,6,0,30.5,25],
[97,1.7,7.7,1,4,26.4,236,16],
[97,1.7,7.7,1,4,5.5,236,16],
[1.2,0.2,2.8,0.4,3.3,2.3,4.9,3.2],
[48,11,31,1,16,4,183,44],
[35,8.5,6,0.6,6,1,136,7.5],
[99,88.1,968,103,18,1,3380.51,88],
[190,72,154,49,158,0,1170,18],
[116,4.2,8,2.5,24.5,1,333,15],
[517,67,1,2,1371,2,168,1],
[48,12,31,1,18,4,183,35],
[46,34,434,18.5,10,8,1373,39],
[528,78,9,3,1342,0,329,9],
[3,0.6,1.5,0.4,8.7,0.9,5.2,0.6],
[119,28,7,2,52,0,430,7],
[118,30,18,7,85,0.5,403,39],
[117,19,13,2,16,20,405,28],
[45.2,21.3,453,32.8,38.9,1,1403,27.2],
[33.5,17.6,192,28.7,14,1,734,6.4],
[70,40,120,8,20,4,335,220],
[12.02,8.7,25.5,2.8,41.7,0.1,103.7,14.2],
[41,3,2,0,2,3,134,3]])
k = np.size(m, axis=0)
n = np.size(m,axis =1)

#Calcul de la matrice centrée réduite
moy=np.average(m,axis=0)
print("moyenne des variables aléatoires : ")
print(moy)
print("variance des vas")
var = np.var(m, axis=0)
print("variance des variables aléatoires : ")
print(var)
##Calcul de la mcr
mcr = m
for j in range (0,n):
    for i in range (0,k):
        mcr[i,j] = (mcr[i,j] - moy[0,j]) / math.sqrt(var[0,j])
print("matrice centrée réduite : ")
print(mcr)

#Matrice des corrélations
correl = 1/k * np.transpose(mcr) * mcr
print("matrice des corrélations : ")
print(correl)

#Valeurs propres et vecteurs propres
(valp,vectp) = np.linalg.eig(correl)
print("les valeurs propres dans l'ordre décroissant")
print(valp)
print("les vecteurs propres correspondants")
print(vectp)
lambda1 = valp[0]
vectp1 = vectp[0]
lambda2 = valp[1]
vectp2 = vectp[1]

#on va travailler avec les 2 premiers vecteurs
#Calculs des coordonnées projetées sur (C1,C2) :
C1 = mcr * np.transpose(vectp1)
C2 = mcr * np.transpose(vectp2)

#Représentation graphique
plt.plot(C2,C1, 'ro')
plt.xlabel("projection selon C1")
plt.ylabel("projection selon C2")
plt.show()

#Calcul du % d'inertie totale exprimée par (C1,C2)
inertieC12= (lambda1+lambda2) / np.sum(np.transpose(valp))
print("% d'inertie expimée par C1 et C2: ")
print(inertieC12)

#Cercle des corrélations

##Initialisation de quelques fonction nécessaires pour calculer la corrélation
def column(matrix, i):
    return np.transpose(matrix)[i]

def cov(X,Y):
    n = np.size(X)
    moyX = np.average(X)
    moyY = np.average(Y)
    s = 0
    for i in range(0,n):
        s = s + (X[i]-moyX) * (Y[i]-moyY)
    return (s/n)

##Traitement :
###on recupere les colonnes des variables aléatoires sous forme d'array
vasShaped = []
for i in range(0,8):
    Xtemp =column(mcr,i)
    vasShaped.append(np.squeeze(np.array(Xtemp)))

## on met C1 et C2 sous la même forme
C1shaped = np.squeeze(np.asarray(C1))
C2shaped = np.squeeze(np.asarray(C2))

##on calcule les corrélations selon C1 et C2 pour chaque variable :
corrC1=[]
corrC2=[]
i = 0
for X in vasShaped:
    corr1 = np.corrcoef(X,C1shaped)[0,1]
    corr2 = np.corrcoef(X,C2shaped)[0,1]
    corrC1.append(corr1)
    corrC2.append(corr2)
    i= i + 1
print(corrC1)
print(corrC2)


#Représentation graphique
plt.plot(corrC2,corrC1, 'ro')
axes = plt.gca()
axes.set_xlim([-1.2,1.2])
axes.set_ylim([-1.2,1.2])
plt.xlabel("C1")
plt.ylabel("C2")
##ajout du cercle
circle1 = plt.Circle((0, 0), 1, color='r', fill= False)
fig = plt.gcf()
ax = fig.gca()
ax.add_artist(circle1)
##ajout du nom des variables aléatoires
for i, txt in enumerate(varNames):
    ax.annotate(txt, (corrC2[i],corrC1[i]))
plt.show()






