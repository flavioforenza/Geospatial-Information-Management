#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:08:58 2019

@author: Flavio Forenza
"""
import statistics
import geopandas as gpd
import matplotlib.pyplot as plt
import math
import pandas as pnd
from shapely.geometry import Point, mapping, Polygon
from sklearn.cluster import KMeans
import numpy as np
import collections
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database, drop_database
from geoalchemy2 import Geometry, WKTElement
from fiona import collection
from fiona.crs import from_epsg
import os
import shutil

path_Directory = '/Users/flavioforenza/Google Drive ISSIA/Google Drive/UNIMI/2 anno/1 semestre/Gig/Progetto GIG/'

tr1 = gpd.read_file(path_Directory+'oneway_meetingA.shp')
tr2 = gpd.read_file(path_Directory+'oneway_meetingB.shp')

fig,ax = plt.subplots(1)
first=tr1.plot(ax=ax, label = 'tr1')
second=tr2.plot(ax=ax, label = 'tr2')
plt.legend(loc='upper left')
#plt.savefig('tr_original_complete')
plt.show()

'''
---------> [NOISE-FILTERING] <---------
'''

def create_DF_Filt(df, tr, count, indexDf, medianaX, medianaY, listTempZ):
    df.loc[indexDf,'tag_id'] = tr.tag_id[count]
    df.loc[indexDf,'time'] = tr.time[count]
    df.loc[indexDf,'epoch'] = tr.epoch[count]
    df.loc[indexDf,'slot'] = tr.slot[count]
    #effettuo ma mediana di tutte le coordinate con lo stesso timestamp
    if(medianaX!=0 or medianaY!=0):
         df.loc[indexDf,'x'] = medianaX
         df.loc[indexDf,'y'] = medianaY
         df.loc[indexDf,'z'] = statistics.median(listTempZ)
         coordinate = [medianaX, medianaY]
    else:
         df.loc[indexDf,'x'] = tr.x[count]
         df.loc[indexDf,'y'] = tr.y[count]
         df.loc[indexDf,'z'] = tr.z[count]
         coordinate = [tr.x[count], tr.y[count]]
    point = gpd.GeoSeries(Point(coordinate))
    df.loc[indexDf,'millisecon'] = tr.millisecon[count]
    df.loc[indexDf,'timestamp'] = tr.timestamp[count]
    return point

#creazione del nuovo Dataframe privo di rumore
def noiseFiltering(tr):
    dimensionTime = range(0, len(tr))
    timestamp1 = pnd.Series(tr.timestamp, index = dimensionTime)
    listX = pnd.Series(tr.x, index = dimensionTime)
    listY= pnd.Series(tr.y, index = dimensionTime)
    listZ = pnd.Series(tr.z, index = dimensionTime)

    #lista temporanea delle x e delle y di ogni secondo
    listTempX = []
    listTempY = []
    listTempZ = []
    
    #creo il nuovo dataFrame per i dati filtrati
    df = pnd.DataFrame({
            'tag_id':[],
            'time':[],
            'epoch':[],
            'slot':[],
            'x':[],
            'y':[],
            'z':[],
            'millisecon':[],
            'timestamp':[]})
    
    indexDf = 0
    count = 0
    listGeometry = []
    for index,row in tr.iterrows():
            if(count>index):
                continue
            #assumo il timestamp
            t1 = row.timestamp
            #se non è l'ultimo indice
            if(index<len(tr)-1):
                flag = True
                count = index
                while(flag):            
                    #acquisisco il nuovo timestamp
                    t2 = timestamp1[count+1]
                    if(t1==t2):
                        #aggiungo le coordinate del primo valore
                        listTempX.append(listX.get(key=count))
                        listTempY.append(listY.get(key=count))
                        listTempZ.append(listZ.get(key=count))
                        count = count + 1
                        if(count==len(tr)-1):
                            flag=False  
                            medianaX = statistics.median(listTempX)
                            medianaY = statistics.median(listTempY)
                            point = create_DF_Filt(df, tr, count, indexDf, medianaX, medianaY, listTempZ)
                            listGeometry.append(point)      
                    else:
                        flag=False
                        if(index!=0):
                            #se non ha gli estremi uguali
                            if(t1!=t2 and t1!=timestamp1[count-1]):
                                #aggiungo il valore ugualmente perchè potrebbe essercene solo 1
                                point = create_DF_Filt(df, tr, count, indexDf, 0, 0, 0)
                                listGeometry.append(point)
                                indexDf = indexDf+1
                        else:
                            #caso in cui il primo elemento è diverso dal secondo
                            if(index==0 and len(listTempX)==0):
                                point= create_DF_Filt(df, tr, count,indexDf, 0, 0, 0)
                                listGeometry.append(point)
                                indexDf = indexDf+1
                        if(len(listTempX)==0):
                            break
                        medianaX = statistics.median(listTempX)
                        medianaY = statistics.median(listTempY)
                        point = create_DF_Filt(df, tr, count, indexDf, medianaX, medianaY, listTempZ)
                        listGeometry.append(point)
                        indexDf = indexDf+1
                #azzero le liste
                listTempX=[]
                listTempY=[]
                listTempZ=[]
    
    df['geometry']= listGeometry
    return df

#assegno le geometrie create (aventi come coordinate le mediane)
dfFilTr1 = noiseFiltering(tr1)
dfFilTr2 = noiseFiltering(tr2)

#elimino i primi 8 secondi dalla seconda traiettoria in quanto inizia prima
list_remove=[]
remove=0
for i in range(0,8):
    list_remove.append(remove)
    remove+=1
dfFilTr2 = dfFilTr2.drop(list_remove)

#reset dell'indice della seconda traiettoria filtrata:
dfFilTr2.reset_index(drop=True, inplace=True)

#Creazione del GeoDataFrame delle 2 traiettorie
gdfTr1 = gpd.GeoDataFrame(dfFilTr1, geometry = gpd.points_from_xy(dfFilTr1.x, dfFilTr1.y))
gdfTr2 = gpd.GeoDataFrame(dfFilTr2, geometry = gpd.points_from_xy(dfFilTr2.x, dfFilTr2.y))

#visualizzazione delle 2 traiettorie filtrate
fig,ax = plt.subplots(1)
gdfTr1.plot(ax=ax, label = 'Filtering tr1')
gdfTr2.plot(ax=ax, label = 'Filtering tr2')
plt.legend(loc='upper left')
#plt.savefig('tr2_filtering')
plt.show()

'''
---------> [PUNTI/ZONE IN COMUNE] <---------
'''

#Calcolo delle distanze
def distance(pls, x2, y2):
    listDistance = []
    dimensionCoord = range(0, len(pls))     
    listGdfX = pnd.Series(pls.x, index = dimensionCoord)
    listGdfY= pnd.Series(pls.y, index = dimensionCoord) 
    for index, row in pls.iterrows():
        x1 = row.x
        y1 = row.y
        #funziona con gli indici ordinati (senza intervalli di distacco)
        if(index<len(pls)-1):
            if x2 is None:
                x3 = listGdfX[index+1]
                y3 = listGdfY[index+1]
                #apply Pythagorean theorem for distance compute
                dist = math.sqrt((x3-x1)**2 + (y3-y1)**2)
                #save distance
                listDistance.append(dist)
            else:
                #apply Pythagorean theorem for distance compute
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                #save distance
                listDistance.append(dist)
    return listDistance

listDistTr1 = distance(gdfTr1, None, None)
listDistTr2 = distance(gdfTr2, None, None)

fig,ax = plt.subplots(1)
box_plot_Tr1=[listDistTr1]
bp1=plt.boxplot(box_plot_Tr1,patch_artist=True, showmeans=True)
#plt.savefig('bp_tr1')
plt.show()

fig,ax = plt.subplots(1)
box_plot_Tr2=[listDistTr2]
bp2=plt.boxplot(box_plot_Tr2,patch_artist=True, showmeans=True)
#plt.savefig('bp_tr2')
plt.show()

media_tr1 = statistics.mean(listDistTr1)
media_tr2 = statistics.mean(listDistTr2)

def sharedArea(bp, listDist):
    #Mediana
    mediana = 0
    count =0
    for i in bp['medians']:
        mediana = i.get_data()[1][0]
    #rimozione outliers
    count =0
    listCount = []
    for value in listDist:
    #se la distaza fra i 2 è < della mediana
        if(value > mediana):
            listCount.append(count+1)
        count = count+1         
    return listCount

gdfFilterTr1 = gdfTr1.copy()
gdfFilterTr1.drop(sharedArea(bp1, listDistTr1), inplace = True)

gdfFilterTr2 = gdfTr2.copy()
gdfFilterTr2.drop(sharedArea(bp2, listDistTr2), inplace = True)

#Visualizzazione dati filtrati
fig,ax = plt.subplots(1, figsize=(10,10))

gdfFilterTr1.plot(ax=ax, label = 'update Tr1')
gdfFilterTr2.plot(ax=ax, label = 'update Tr2')

plt.legend(loc='upper left')
#plt.savefig('update.png')

#reset dell'indice 
#gdfFilter mantiene il suo indice originale
gdfCopyTempTr1 = gdfFilterTr1.copy()
gdfCopyTempTr1.reset_index(drop=True, inplace=True)

gdfCopyTempTr2 = gdfFilterTr2.copy()
gdfCopyTempTr2.reset_index(drop=True, inplace=True)

'''
---------> [K-MEANS] <---------
'''

#individuo i cluster sse n_punti>1
def clusterDetection(gdfCopyTemp):
    listDistKnn=[]
    numCluster =0
    for index, rows in gdfCopyTemp.iterrows():
        geom = gdfCopyTemp.loc[index].geometry
        if(index<len(gdfCopyTemp)-1):
            cc = geom.distance(gdfCopyTemp.loc[index+1].geometry)
            listDistKnn.append(cc)
            if(len(listDistKnn)>0):
                #considero un cluster solo se ha più di 1 punto
                if (abs(cc-listDistKnn[index-1])>1):
                    numCluster+=1
     
    #array contenente le coordinate dei dati filtrati       
    arrayX=gdfCopyTemp.x.values
    arrayY=gdfCopyTemp.y.values
    
    #dataframe utile per il KNN
    df = pnd.DataFrame({
            'x': arrayX,
            'y': arrayY
            })
    
    return df, numCluster

def K_MEANS(dfTr, numCluster):
    kmeans = KMeans(numCluster-1)
    kmeans.fit(dfTr)
    #caclolo delle lables (N punti) che formano ogni cluster
    labels = kmeans.predict(dfTr)
    return kmeans, labels
    
#applicazione del k-means
dfTr1, nCluster_Tr1= clusterDetection(gdfCopyTempTr1)
kmeans_tr1, labels_tr1 = K_MEANS(dfTr1, nCluster_Tr1) 

dfTr2, nCluster_Tr2= clusterDetection(gdfCopyTempTr2)
kmeans_tr2, labels_tr2 = K_MEANS(dfTr2, nCluster_Tr2) 

#conteggio delle labels in modo da rimuovere quelle meno popolate
dictCount_tr1 = collections.Counter(labels_tr1)
dictCount_tr2 = collections.Counter(labels_tr2)

#rimuove le labels avente un conteggio = 1 (1 solo punto)
def controlLabels(dictCount, numCluster):
    listDelete = []
    for k,v in dictCount.items():
        if(v==1):
            #trovo la label da elminimare
            listDelete.append(v-1)
    #aggiorno il numero dei cluster:
    for i in range(len(listDelete)):
        numCluster-=1
    return numCluster, listDelete
        
#rimuovo il punto dal dataframe
nCluster_Tr1, listDelete_Tr1 = controlLabels(dictCount_tr1, nCluster_Tr1)
dfTr1 = dfTr1.drop(listDelete_Tr1)

nCluster_Tr2, listDelete_Tr2 = controlLabels(dictCount_tr2, nCluster_Tr2)
dfTr2 = dfTr2.drop(listDelete_Tr2)

#ripeto il kmeans
kmeans_tr1, labels_tr1 = K_MEANS(dfTr1, nCluster_Tr1) 
kmeans_tr2, labels_tr2 = K_MEANS(dfTr2, nCluster_Tr2) 
    
#calcolo dei centroidi                
centroids_tr1 = kmeans_tr1.cluster_centers_
centroids_tr2 = kmeans_tr2.cluster_centers_
#plt.show()

#creo un buffer per ogni centroide ma per farlo devo
#creare i punti con le coordinate dei centroidi
def pointCentroid(centroids):
    listPoints=[]
    for i in range(len(centroids)):
        listPoints.append(Point((centroids[i])))
    return listPoints

#conteggio del numero di punti in ogni cluster
listLables_Tr1 = labels_tr1.tolist() 
elem_tr1= [listLables_Tr1.count(i) for i in range(max(listLables_Tr1)+1)]

#calcolo distanza punti dal proprio centoride del cluster
def distance_Point_Centroid(coordinate,init, fine, df):
    centroid = Point(coordinate)
    distToCentr=[]
    if(fine!=len(df)):
        fine+=1
    for i in range(init, fine):
        coordinate = [df.x[i],df.y[i]]
        point = Point(coordinate) 
        distance = point.distance(centroid)
        distance = distance**2
        distToCentr.append(distance)
    return distToCentr

#liste delle distanze al quadrato di ogni punto dal proprio centroide (Tr1)
listFirstCentroid_tr1 = []
listFirstCentroid_tr1.append(distance_Point_Centroid((50.9171, 24.3319),1, 7, dfTr1))
listSecondCentroid_tr1 = []
listSecondCentroid_tr1.append(distance_Point_Centroid((23.7198, 1.33712),8, 32, dfTr1))
listThirdCentroid_tr1 = []
listThirdCentroid_tr1.append(distance_Point_Centroid((15.3206, 3.78145),33,111, dfTr1))  

#liste delle distanze al quadrato di ogni punto dal proprio centroide (Tr2)
listFirstCentroid_tr2 = []
listFirstCentroid_tr2.append(distance_Point_Centroid((50.0375, 24.4358),0, 18, dfTr2))
listSecondCentroid_tr2 = []
listSecondCentroid_tr2.append(distance_Point_Centroid((23.7236, 1.96597),19, 41, dfTr2))
listThirdCentroid_tr2 = []
listThirdCentroid_tr2.append(distance_Point_Centroid((15.6395, 3.87138),42,111, dfTr2)) 
 
#devo sommare fra di loro le distanze
def sumListDist(listCentr):
    total = 0
    for ele in range(len(listCentr)):
        total = total + listCentr[:,ele] 
    return total    

#somme delle distanze al quadrato
Tr1sum_dist1_square=sumListDist(np.array(listFirstCentroid_tr1))
Tr1sum_dist2_square=sumListDist(np.array(listSecondCentroid_tr1))
Tr1sum_dist3_square=sumListDist(np.array(listThirdCentroid_tr1))

Tr2sum_dist1_square=sumListDist(np.array(listFirstCentroid_tr2))
Tr2sum_dist2_square=sumListDist(np.array(listSecondCentroid_tr2))
Tr2sum_dist3_square=sumListDist(np.array(listThirdCentroid_tr2))

#somma delle distanze al quadrato di tutti i cluster di una sola traiettoria
#SSE
SSE_tr1= Tr1sum_dist1_square+Tr1sum_dist2_square+Tr1sum_dist3_square
SSE_tr2= Tr2sum_dist1_square+Tr2sum_dist2_square+Tr2sum_dist3_square

listLables_Tr2 = labels_tr2.tolist()   
elem_tr2 = [listLables_Tr2.count(i) for i in range(max(listLables_Tr2)+1)]

#creo i punti per i nuovi centroidi
gdfCentroids_tr1 = gpd.GeoDataFrame(geometry = pointCentroid(centroids_tr1))
gdfCentroids_tr2 = gpd.GeoDataFrame(geometry = pointCentroid(centroids_tr2))

def printBuff(gdfCentroids_tr):
    for i in range(len(gdfCentroids_tr1)):
        #creo il buffer per ogni punto centroide specificando la dimensione 
        gdfCentroids_tr.geometry[i] = gdfCentroids_tr.geometry[i].buffer(1, resolution = 16)
    return gdfCentroids_tr 
   
#assegno i buffer creati sostituendo i punti raffiguranti i centroidi
gdfCentroids_tr1 = printBuff(gdfCentroids_tr1)
gdfCentroids_tr2 = printBuff(gdfCentroids_tr2)

#Visualizzazione dati filtrati
fig,ax = plt.subplots(1)

def pltCentr(centroids, color, m):
    for idx, centroid in enumerate(centroids):
        a = plt.scatter(*centroid, c = color, marker= m) 
    return a  

#stampo i centroidi con i corispettivi buffer        
tr1_c = pltCentr(centroids_tr1, "blue", ".")

gdfCentroids_tr1.plot(ax = ax, color='blue')
tr2_c = pltCentr(centroids_tr2, "orange", ".")
gdfCentroids_tr2.plot(ax=ax, color='orange')
#traiettorie filtrate
gdtr1= gdfTr1.plot(ax=ax, alpha=0.3)
gdtr2= gdfTr2.plot(ax=ax, alpha=0.3)
plt.legend((tr1_c, tr2_c, gdtr1, gdtr2),('Buf_tr1', 'Buf_tr2'),loc='upper left')
#plt.legend(loc='upper left')


##punti di ogni cluster (di ogni traiettoria)
#plt.scatter(dfTr1['x'], dfTr1['y'], alpha=0.5)
#plt.scatter(dfTr2['x'], dfTr2['y'], alpha=0.5)

def printLabel(numCluster, centroids_tr, number, box, arrow):
    num = []
    for i in range(0, numCluster-1):
        num.append(i+1)
    xC = centroids_tr[:,0]
    yC = centroids_tr[:,1]
    for i, txt in enumerate(num):
        ax.annotate(txt, (xC[i], yC[i]), color=number, textcoords='offset pixels', ha='center', 
                    va='bottom',bbox=dict(boxstyle='round,pad=0.6', fc=box),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.95', color=arrow))
 
#printLabel(nCluster_Tr1, centroids_tr1, 'yellow', 'blue', 'red') 
#printLabel(nCluster_Tr2, centroids_tr2, 'red', 'orange', 'green') 
   
#plt.savefig('NumbClusterTr2')
plt.show()

#determino il tempo trascorso in ogni area di stop
#vedo quali punti cadono all'interno del buffer
def labes_unique(labels):
    arr_lab = np.array(labels)
    indexes = np.unique(arr_lab, return_index=True)[1]
    #faccio ritornare la lista con l'ordine del loro indice
    list_l =[arr_lab[index] for index in sorted(indexes)]
    #arr_lab_tmp = np.unique(arr_lab, return_index=True)
    return list_l

unique_tr1 = labes_unique(listLables_Tr1)
unique_tr2 = labes_unique(listLables_Tr2)

#spazi attraversati in comune
fig,ax = plt.subplots(1)
#l'intersezione riguarda solo le aree in comune fra i buffer
intersections = gpd.overlay(gdfCentroids_tr1,gdfCentroids_tr2, how='intersection')
intersections.plot(ax=ax, alpha=0.5, edgecolor='k', cmap='tab10');
#plt.savefig('Intersection')
plt.show()

'''
---------> [INTERPOLATION] <---------
(Solo per la prima traiettoria)
'''

#prendo i punti che hanno la distanza massima
p1_tr1 = listDistTr1.index(max(listDistTr1))

def interpolate(gdfTr, index, count):
    listNumberX = []
    listNumberY= []
    listNumberX.append(gdfTr.x[index])
    listNumberY.append(gdfTr.y[index])
        
    for i in range(count):
        listNumberX.append(np.nan)
        listNumberY.append(np.nan)
        
    listNumberX.append(gdfTr.x[index+1])
    listNumberY.append(gdfTr.y[index+1])

    interpSeriesX = pnd.Series(listNumberX)
    pointInterpX = interpSeriesX.interpolate()
    
    interpSeriesY = pnd.Series(listNumberY)
    pointInterpY = interpSeriesY.interpolate()
    
    #creo un nuovo dataframe
    dfTemp = pnd.DataFrame({
            'x': pointInterpX,
            'y': pointInterpY
            })
    
    #calcolo le distanze fra i punti
    gdf = gpd.GeoDataFrame(dfTemp, geometry = gpd.points_from_xy(dfTemp.x, dfTemp.y))
    listDist_interp = distance(gdf, None, None) 

    return listDist_interp, dfTemp

points_missing = 1
gdf_interp_tr1 = gdfTr1.copy()
listDist, df = interpolate(gdf_interp_tr1, p1_tr1, points_missing)
media = statistics.mean(listDist)
#media_tr1
while(media>(media_tr1+0.5)):
    points_missing+=1
    listDist, dfTemp = interpolate(gdf_interp_tr1, p1_tr1, points_missing)
    media = statistics.mean(listDist)

#inserisco i nuovi punti nel dataframe iniziale
df1_interp = pnd.concat([gdf_interp_tr1.iloc[:p1_tr1], dfTemp, gdf_interp_tr1.iloc[p1_tr1+1:]], sort=False).reset_index(drop=True)

fig,ax = plt.subplots(1)
first=df1_interp.plot(ax=ax, label = 'Interpolation tr1')
##second=tr2.plot(ax=ax, label = 'tr2')
plt.legend(loc='upper left')
#plt.savefig('interp_final')
plt.show()


'''
---------> [CAMMINI] <---------
(Cammini vicini e distanti fra le 2 traiettorie)
'''

#Determino la distanza di ogni individuo allo nello stesso timestamp
list_pointDist=[]
for index, row in gdfTr1.iterrows():
    geomTr1 = gdfTr1.loc[index].geometry
    geomTr2 = gdfTr2.loc[index].geometry 
    distance= geomTr1.distance(geomTr2)
    list_pointDist.append(distance)

#boxplot delle distanze
fig,ax = plt.subplots(1)
box_plot_walk=[list_pointDist]
bpWalk=plt.boxplot(box_plot_walk,patch_artist=True, showmeans=True)
#plt.savefig('bpCammini_tr1')
plt.show()

arrDist = pnd.Series(list_pointDist)
q1 = arrDist.quantile(q=0.25)
q3 = arrDist.quantile(q=0.75)

iqr = q3 - q1

outliers = []
for value in arrDist:
    if value > q3 + 1.5*iqr or value < q1 - 1.5*iqr:
        outliers.append(value)

#identificazione punti con distanza >= a min(outliers)
#dataframe utile per gli outliers
    
def get_dataFrame():
    df = pnd.DataFrame({
            'x': [],
            'y': [],
            'timestamp':[]
            })
    return df
    
#creo un df per i percorsi distanti e vicini
df_out1 = get_dataFrame()
df_in1= get_dataFrame()

df_out2 = get_dataFrame()
df_in2= get_dataFrame()

geometry_out1=[]
geometry_out2=[]
geometry_in1=[]
geometry_in2=[]

count =0
count_dist=0
count_near=0
for index, row in gdfTr1.iterrows():
    geomTr1 = gdfTr1.loc[count].geometry
    geomTr2 = gdfTr2.loc[count].geometry 
    distance= geomTr1.distance(geomTr2)
    #traiettorie distanti
    if distance>=min(outliers):
        df_out1.loc[count_dist,'x'] = gdfTr1.x[count]
        df_out1.loc[count_dist,'y'] = gdfTr1.y[count]
        df_out1.loc[count_dist,'timestamp'] = gdfTr1.timestamp[count]
        df_out2.loc[count_dist,'x'] = gdfTr2.x[count]
        df_out2.loc[count_dist,'y'] = gdfTr2.y[count]
        df_out2.loc[count_dist,'timestamp'] = gdfTr2.timestamp[count]
        geometry_out1.append(geomTr1)
        geometry_out2.append(geomTr2)
        count_dist+=1
    #traiettorie vicine
    else:
        df_in1.loc[count_near,'x'] = gdfTr1.x[count]
        df_in1.loc[count_near,'y'] = gdfTr1.y[count]
        df_in1.loc[count_near,'timestamp'] = gdfTr1.timestamp[count]
        df_in2.loc[count_near,'x'] = gdfTr2.x[count]
        df_in2.loc[count_near,'y'] = gdfTr2.y[count]
        df_in2.loc[count_near,'timestamp'] = gdfTr2.timestamp[count]
        geometry_in1.append(geomTr1)
        geometry_in2.append(geomTr2)
        count_near+=1
    count+=1

#aggiungo la geometria al df per le traiettorie distanti
df_out1['geometry']= geometry_out1
df_out2['geometry']= geometry_out2

#aggiungo la geometria al df per le traiettorie vicine
df_in1['geometry']= geometry_in1
df_in2['geometry']= geometry_in2


#Creazione del GeoDataFrame delle 2 traiettorie distanti
gdfTr1_out = gpd.GeoDataFrame(df_out1, geometry = gpd.points_from_xy(df_out1.x, df_out1.y))
gdfTr2_out= gpd.GeoDataFrame(df_out2, geometry = gpd.points_from_xy(df_out2.x, df_out2.y))
       
#visualizzazione delle 2 traiettorie distanti
fig,ax = plt.subplots(1)
gdfTr1_out.plot(ax=ax, label = 'Traiettoria 1')
gdfTr2_out.plot(ax=ax, label = 'Traiettoria 2')
plt.legend(loc='upper left')
#plt.savefig('Distanti')
plt.show() 

#Creazione del GeoDataFrame delle 2 traiettorie vicine
gdfTr1_in = gpd.GeoDataFrame(df_in1, geometry = gpd.points_from_xy(df_in1.x, df_in1.y))
gdfTr2_in= gpd.GeoDataFrame(df_in2, geometry = gpd.points_from_xy(df_in2.x, df_in2.y))

#visualizzazione delle 2 traiettorie vicine
fig,ax = plt.subplots(1)
gdfTr1_in.plot(ax=ax, label = 'Traiettoria 1')
gdfTr2_in.plot(ax=ax, label = 'Traiettoria 2')
plt.legend(loc='upper left')
#plt.savefig('Vicini')
plt.show()   

'''
---------> [CREAZIONE E SALVATAGGIO (SHAPEFILE) IN LOCALE] <---------
(Per ogni gdf viene creato un shapefile nella cartella di lavoro)
'''

def create_Shape(path, nameFile, df, geom):
    #creo un nuovo shapefile
    schema = { 'geometry': geom, 'properties': { } }
    #creazione della cartella
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        #rimuovo le sotto-directory
        shutil.rmtree(path)           
        os.makedirs(path)
    with collection(
        path+nameFile, "w", "ESRI Shapefile", 
        schema, crs=from_epsg(3003)) as output:
        for i in gdf.geometry:
            if(geom=='Point'):
                output.write({'properties': {}, 
                              'geometry': mapping(Point(i.x, i.y))})
            else:
                output.write({'properties': {}, 
                              'geometry': mapping(Polygon(i))})

path_save= r"/Users/flavioforenza/Google Drive ISSIA/Google Drive/UNIMI/2 anno/1 semestre/Gig/Progetto GIG/"

def local_shape(nameFile, nameDir, df, geom):
    nameF = nameFile
    nameD = nameDir
    create_Shape(path_save+nameD, nameF, df, geom)

#Restituisce il df con la colonna geometry
def create_dfGeom(df):
    list_points=[]
    for index, row in df.iterrows():
        point = Point(row[0], row[1])
        list_points.append(point)
    return list_points

#1 GeoDataframe: Traiettoria n°1 filtrata
local_shape("/Filter_tr1.shp", "Tr-Filter", dfFilTr1, 'Point')
#points_df = gpd.read_file(path+nameDir+name_File)
#2 GeoDataframe: Traiettoria n°2 filtrata
local_shape("/Filter_tr2.shp", "Tr-Filter", dfFilTr2, 'Point')
#3. K-Means Traiettoria n°1
dfTr1['geometry'] = create_dfGeom(dfTr1)
local_shape("/K-meeans_tr1.shp", "K-Means", dfTr1, 'Point')
#4. K-Means Traiettoria n°2
dfTr2['geometry'] = create_dfGeom(dfTr2)
local_shape("/K-meeans_tr2.shp", "K-Means", dfTr2, 'Point')
#5. Buffers Traiettoria 1
local_shape("/Centroids_tr1.shp", "Buffer", gdfCentroids_tr1, 'Polygon')
#6. Buffers Traiettoria 2
local_shape("/Centroids_tr2.shp", "Buffer", gdfCentroids_tr2, 'Polygon')
#7. Interpolazione Traiettoria n°1
local_shape("/Interpolation_tr1.shp", "Interpolazione", df1_interp, 'Point')
#8. Cammino Distante Traiettoria n°1
local_shape("/Walk_Distance_tr1.shp", "Cammini", gdfTr1_out, 'Point')
#9. Cammino Distante Traiettoria n°2
local_shape("/Walk_Distance_tr2.shp", "Cammini", gdfTr2_out, 'Point')
#10. Cammino Vicino Traiettoria n°1
local_shape("/Walk_Near_tr1.shp", "Cammini", gdfTr1_in, 'Point')
#11. Cammino Vicino Traiettoria n°2
local_shape("/Walk_Near_tr2.shp", "Cammini", gdfTr2_in, 'Point')


'''
---------> [SALVATAGGIO DATI POSTGRE-SQL] <---------
'''

#credenziali
POSTGRESQL_USER='postgres'
POSTGRESQL_PASSWORD='password'
POSTGRESQL_HOST_IP='127.0.0.1'
POSTGRESQL_PORT='5432'
POSTGRESQL_DATABASE='Progetto Forenza'

#connessione 
try:
    engine = create_engine('postgresql://'+POSTGRESQL_USER+
                           ':'+POSTGRESQL_PASSWORD+
                           '@'+POSTGRESQL_HOST_IP+
                           ':'+POSTGRESQL_PORT+
                           '/'+POSTGRESQL_DATABASE,echo=False)
    if database_exists(engine.url):
        drop_database(engine.url)
        create_database(engine.url)
    else:
        create_database(engine.url)
        
    print('Connessione PostgreSQL avvenuta con successo!')
except:
    print ("Errore! Non è stato possibile connettersi a PostgreSQL.")
    print ("Verificare che il Server sia funzionante")
    
conn = engine.connect()

# Creazione estensione postgis (utile per inserimento colonna geometria)
trans = conn.begin()
conn.execute('CREATE EXTENSION postgis')
trans.commit()

def Postgres_SQL(df,name_table, typeGeom):
    #assegno tutti i dati tranne originali comprese le geometrie
    my_geo_df = gpd.GeoDataFrame(df)
    #conversione geometrie  in formato WKT
    my_geo_df['geom'] = my_geo_df['geometry'].apply(lambda x: WKTElement(x.wkt, srid=3003))
    
    #rimuvo la colonna geometry in quanto ora è duplicata
    my_geo_df.drop('geometry', 1, inplace=True)
    
    #passo l'intero GeodataFrame a PostgreSql
    my_geo_df.to_sql(name=name_table, con=engine, if_exists = 'replace', index=False,
                     dtype={'geom': Geometry(typeGeom, srid= 3003)})

#DATAFRAME MERORIZZATI
#1. Traiettoria n°1 Filtrata    
Postgres_SQL(dfFilTr1, 'Filter_tr1', 'Point') 
#2. Traiettoria n°2 Filtrata    
Postgres_SQL(dfFilTr2, 'Filter_tr2', 'Point') 
#3. K-Means Traiettoria n°1
dfTr1['geometry'] = create_dfGeom(dfTr1)
Postgres_SQL(dfTr1, 'K-Means Tr1', 'Point')
#4. K-Means Traiettoria n°2
dfTr2['geometry'] = create_dfGeom(dfTr2)
Postgres_SQL(dfTr2, 'K-Means Tr2', 'Point')
#5. Centroidi Traiettoria 1
Postgres_SQL(gdfCentroids_tr1, 'Centroidi_Tr1', 'Polygon')
#6. Centroidi Traiettoria 2
Postgres_SQL(gdfCentroids_tr2, 'Centroidi_Tr2', 'Polygon')
#7. Interpolazione Traiettoria n°1
Postgres_SQL(df1_interp, 'Interpolazione Tr1', 'Point')
#8. Cammino Distante Traiettoria n°1
Postgres_SQL(gdfTr1_out, 'Cammmino Tr1 Distante', 'Point')
#9. Cammino Distante Traiettoria n°2
Postgres_SQL(gdfTr2_out, 'Cammmino Tr2 Distante', 'Point')
#10. Cammino Vicino Traiettoria n°1
Postgres_SQL(gdfTr1_in, 'Cammmino Tr1 Vicino', 'Point')
#11. Cammino Vicino Traiettoria n°2
Postgres_SQL(gdfTr2_in, 'Cammmino Tr2 Vicino', 'Point')
