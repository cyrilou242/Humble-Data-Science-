import matplotlib.pyplot as plt
import numpy as np
from numba import jit

np.set_printoptions(precision=4, suppress=False)

"""
Exercice KMeans
Cyril de Catheu
Last update 21/02/2018
"""

"""
#############################################################################
############################## CODAGE DU KMEANS #############################
#############################################################################
"""

# on défini un loader de jeu de données
def loader(filepath):
    """Chargement d'un jeu de données, assumé par la suite de dimension 2"""
    data = np.loadtxt(filepath)
    return data

@jit
def distanceEucl(a, b):
    """Calcul de la distance euclidienne en dimension quelconque"""
    dist = np.linalg.norm(a - b)
    return dist

#@jit
def Kmeans(k, data):
    """Classification par la méthode Kmeans naive, renvoie la liste des classes et les centres correspondants"""
    len = np.size(data, axis=0);
    nb_var = np.size(data, axis=1);
    # on choisi k centre au hasard dans le jeu de données
    idx = np.random.randint(len, size=k)
    centres1 = data[idx, :]
    # initialisation du centre(n-1) avec des 0
    # assume que le centre obtenu précedemment est différent
    centres0 = np.ones((k, nb_var))
    # boucle principale
    while not (np.absolute(centres1 - centres0) < 1e-3).all():
        #initialisation de la forme de la liste des groupes:
        groups_list = []
        for i in range(0, k):
            groups_list.append([])
        # on réparti dans les groupes
        for val in data:
            distanceCentres = []
            compteur = 0
            for c in centres1:
                distanceCentres.append((compteur, distanceEucl(c, val)))
                compteur = compteur + 1
            # coordonnées du meilleur centre
            meilleurCentre = 0
            for i in range(1, k):
                if (distanceCentres[meilleurCentre][1] > distanceCentres[i][1]):
                    meilleurCentre = i
            # on ajoute la valeur au groupe
            groups_list[meilleurCentre].append(val)
            # on stocke les anciens centre
        centres0 = np.matrix(centres1)
        # on calcule les nouveaux centres
        for i in range(0, k):
            average = np.average(np.array(groups_list[i]), axis=0)
            centres1[i, :] = average

    return groups_list, centres1

@jit
def inerGroupe(groupe, centre):
    """Calcul de l'inertie entre 2 groupes"""
    iner = 0
    nbElements = len(groupe)
    for i in range(0, nbElements):
        iner += distanceEucl(groupe[i], centre) ** 2
    return iner

@jit
def inerIntra(groupes, centres):
    """Calcul de l'inertie intra-groupe d'une liste de groupes"""
    inerGroupeList = []
    i = 0
    for centre in centres:
        inerGroupeList.append(inerGroupe(groupes[i], centre))
        i = i + 1
    inersum = 0
    for iner in inerGroupeList:
        inersum += iner
    return inersum

@jit
def inerInter(groupes, centres):
    """Calcul de l'inertie inter-groupes d'une liste de groupes"""
    # calcul du centre global
    listeTot = []
    for groupe in groupes:
        listeTot += groupe
    centreTot = np.average(np.array(listeTot), axis=0)

    inersum2 = 0
    i = 0
    for centre in centres:
        nb_elements = len(groupes[i])
        inersum2 += nb_elements * (distanceEucl(centreTot, centre) ** 2)
        i = i + 1
    return inersum2

@jit
def KmeansIntraCompare(k, data, nbTests):
    """Réalisation d'un nombre donné de classification Kmeans.
    Le meilleur résultat selon le critère d'inertie intra-groupe est affiché"""
    KmeansResults = []
    for i in range(0, nbTests):
        KmeansResults.append(Kmeans(k, data))

    # on réduit l'inertie intra-groupe donc on privilégie des groupes homogènes
    best_kmeans = 0
    for i in range(1, nbTests):
        if inerIntra(KmeansResults[best_kmeans][0], KmeansResults[best_kmeans][1]) > inerIntra(KmeansResults[i][0], KmeansResults[i][1]):
            best_kmeans = i
    return KmeansResults[best_kmeans]

@jit
def KmeansInterCompare(k, data, nbTests):
    """Réalisation d'un nombre donné de classification Kmeans.
    Le meilleur résultat selon le critère d'inertie inter-groupe est affiché"""
    KmeansResults = []
    for i in range(0, nbTests):
        KmeansResults.append(Kmeans(k, data))

    # on maximise l'inertie inter-groupe donc on privilégie la séparation des groupes
    best_kmeans = 0
    for i in range(1, nbTests):
        if inerInter(KmeansResults[best_kmeans][0], KmeansResults[best_kmeans][1]) < inerInter(KmeansResults[i][0], KmeansResults[i][1]):
            best_kmeans = i
    return KmeansResults[best_kmeans]

@jit
def colorPick(i):
    """Renvoie une couleur et le typage de point pour une classe à afficher par plot"""
    if i % 6 == 0:
        color = 'b.'
    elif i % 6 == 1:
        color = 'g.'
    elif i % 6 == 2:
        color = 'k.'
    elif i % 6 == 3:
        color = 'm.'
    elif i % 6 == 4:
        color = 'c.'
    else:             # i%6 == 5
        color = 'y.'
    return color

@jit
def represent(groupes, centres):
    """Représente dans le plan une liste de classe dans des couleurs distinctes
    et représente une liste de centres de manière plus visible"""
    nb_groups = len(groupes)
    for i in range(0, nb_groups):
        x = []
        y = []
        for e in groupes[i]:
            x.append(e[0])
            y.append(e[1])
        plt.plot(x, y, colorPick(i))
    for i in range(0, nb_groups):
        # dans une autre boucle pour que les barycentre apparaissent bien au dessus des autres points
        plt.plot(centres[i, 0], centres[i, 1], 'ro')
    return plt.show()

"""
#############################################################################
#################### CODAGE CLASSIFICATION HIERARCHIQUE #####################
#############################################################################
"""

@jit
def sautMini(group1, group2):
    """Critère du saut minimal"""
    mini = float("inf")
    for e in group1:
        for t in group2:
            dist = distanceEucl(e,t)
            if dist < mini:
                mini = dist
    return mini

@jit
def sautMaxi(group1, group2):
    """Critère du saut maximal"""
    maxi = 0
    for e in group1:
        for t in group2:
            dist = distanceEucl(e,t)
            if dist > maxi:
                maxi = dist
    return maxi

@jit
def distMoy(group1, group2):
    """Critère de la distance moyenne"""
    sum = 0
    for e in group1:
        for t in group2:
            sum += distanceEucl(e,t)
    nb_elements1 = len(group1)
    nb_elements2 = len(group2)
    dist = sum / (nb_elements1 * nb_elements2)
    return dist


@jit
def classAsc(k, data, crit):
    """"Classification ascendante naive pour un critère donné"""
    groups_list = []
    for val in data:
        groups_list.append([val])
    while len(groups_list) > k:
        #affichage de l'avancement:
        if len(groups_list) % 10 == 0:
            print("listes restantes :")
            print(len(groups_list))
        # calcul de la matrice de dissimilarité, on prend les coordonnées du minimum au passage
        matDissim = np.zeros((len(groups_list), len(groups_list)))
        i_min, j_min = 0, 1
        mini = float("inf")
        for i in range(0,len(groups_list)):
            for j in range(i+1,len(groups_list)):
                dissim = crit(groups_list[i], groups_list[j])
                matDissim[i,j] = dissim
                if mini >= dissim:
                    mini = dissim
                    i_min, j_min = i, j
        #fusion des deux groupes
        for val in groups_list[j_min]:
            groups_list[i_min].append(val)
        del groups_list[j_min]
    #calculs des centres pour avoir le même output que la fonction Kmeans
    nb_var = np.size(data, axis=1);
    centres = np.ones((k, nb_var))
    for i in range(0, k):
        average = np.average(np.array(groups_list[i]), axis=0)
        centres[i, :] = average

    return groups_list,centres

@jit
def fastClassAsc(k,data):
    """"Classification ascendante optimisée avec le critère des distances moyennes appliquées aux barycentres"""
    groups_list = []
    groups_bary = []
    groups_effectifs =[]
    for val in data:
        groups_list.append([val])
        groups_bary.append(val)
        groups_effectifs.append(1)
    while len(groups_list) > k:
        #affichage de l'avancement:
        if len(groups_list) % 10 == 0:
            print("listes restantes :")
            print(len(groups_list))
        # calcul de la matrice de dissimilarité, on prend les coordonnées du minimum au passage
        matDissim = np.zeros((len(groups_list), len(groups_list)))
        i_min, j_min = 0, 1
        mini = float("inf")
        for i in range(0,len(groups_list)):
            for j in range(i+1,len(groups_list)):
                ratio = (groups_effectifs[i] * groups_effectifs[j])/(groups_effectifs[i] + groups_effectifs[j])
                dissim = ratio * (distanceEucl(groups_bary[i],groups_bary[j])**2)
                matDissim[i,j] = dissim
                if mini >= dissim:
                    mini = dissim
                    i_min, j_min = i, j
        #fusion des deux groupes
        for val in groups_list[j_min]:
            groups_list[i_min].append(val)
        effectif = groups_effectifs[i_min] + groups_effectifs[j_min]
        barys = np.array([groups_bary[i_min], groups_bary[j_min]])
        new_bary = np.average(barys, axis=0, weights=[groups_effectifs[i_min]/effectif,groups_effectifs[j_min]/effectif])
        groups_bary[i_min] = new_bary
        groups_effectifs[i_min] += groups_effectifs[j_min]
        del groups_list[j_min]
        del groups_bary[j_min]
        del groups_effectifs[j_min]
    #Trans-typage de groups_bary pour avoir le même output que les fonctions précédentes
    centres = np.array(groups_bary)

    return groups_list,centres


"""
#############################################################################
############################## DEBUT DES TESTS ##############################
#############################################################################
"""

data1 = loader("TPkmeans1.txt")
data2 = loader("TPkmeans2.txt")

to_print= []

# """Tests des Kmeans"""
# #pour data1
# print("Test Kmeans pour data1...")
# to_print.append(Kmeans(2,data1))
# to_print.append(Kmeans(2,data1)) #ca peut varier
# to_print.append(Kmeans(2,data1)) #ca peut varier encore
#
to_print.append(Kmeans(4,data1))
#to_print.append(Kmeans(4,data1)) #ca peut varier
#to_print.append(Kmeans(4,data1)) #ca peut varier
#
# to_print.append(KmeansIntraCompare(2,data1, 8)) #le meilleur de 8 itérations intra
# to_print.append(KmeansInterCompare(2,data1, 8)) #le meilleur de 8 itérations inter
#
# #pour data2
# print("Test Kmeans pour data2...")
#to_print.append(Kmeans(2,data2))
#to_print.append(Kmeans(2,data2)) #ca peut varier
#to_print.append(Kmeans(2,data2)) #ca peut varier encore
#

#to_print.append(Kmeans(4,data2))
#to_print.append(Kmeans(4,data2)) #ca peut varier
#to_print.append(Kmeans(4,data2)) #ca peut varier encore

#to_print.append(KmeansIntraCompare(4,data2, 8)) #le meilleur de 8 itérations intra
#to_print.append(KmeansInterCompare(4,data2, 8)) #le meilleur de 8 itérations inter

"""Tests de la classification ascendante"""

# #pour data1
# print("Test CHA pour data1...")
# to_print.append(classAsc(2,data1,sautMini))
# to_print.append(classAsc(2,data1,sautMaxi))
# to_print.append(classAsc(2,data1,distMoy))
# to_print.append(fastClassAsc(2,data1))

#pour data2 : on n'utilise pas l'algo naif, trop long
#print("Test CHA pour data2...")
#to_print.append(fastClassAsc(4,data2))

"""Affichage des résultats"""
for tup in to_print:
    represent(tup[0],tup[1])