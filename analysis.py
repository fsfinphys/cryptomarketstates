import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
import math

from sklearn.manifold import MDS
from sklearn.cluster import KMeans

from data_collection import get_data, load_crypto_names

def analysisdata(symbols, startdate, enddate, tinterval):
    series = dict()
    for i in symbols:
        series[i] = get_data(i, start_date=startdate, end_date=enddate, index_as_date=True, interval=tinterval)['close']
    return series

def rmatrices(startdate, enddate, epoch, cryptosymbols=[]):
    """
    get n R matrices for our crypto symbols, input a start/end date as a string 'mdy', input epoch in integer days
    function calculates number n of matrices it will make (be careful must check it is an integer beforehand)
    initializes an empty array
    loops through starting at the original start date and gives a dictionary which we turn into a dataframe
    once a dataframe we perform some math operators and drop the first row (returns NaN values)
    then becomes R dataframe (matrix) and we append it to our array
    Does this for the entire data time and then returns the list of all the matrices
    note; here we make our R matrix have 1 day time intervals between data points 
    (not 1wk like we did above to determine symbols)
    note; yahoo fin requires at least a 1 month interval to pull data for some reason?
    """
    if len(cryptosymbols) == 0:
        cryptosymbols = load_crypto_names()
    
    n = int((datetime.strptime(enddate, '%m%d%y') - datetime.strptime(startdate, '%m%d%y')).days/epoch)
    matrices = list()
    print('expected number of matrices:', n)
    for i in range(n):
        tempdict = analysisdata(cryptosymbols, datetime.strftime(datetime.strptime(startdate, '%m%d%y') + timedelta(days=(i*epoch)), '%m%d%y'), datetime.strftime(datetime.strptime(startdate, '%m%d%y') + timedelta(days=((i+1)*epoch)), '%m%d%y'), tinterval= '1d')
        tempdf = pd.DataFrame.from_dict(tempdict)
        tempdftwo = np.log(tempdf).diff()
        r = tempdftwo.drop(index=tempdftwo.index[0])
        matrices.append(r)
    return matrices

def noisesupression(corrmatrices, e):
    """
    function which does power mapping for noise supresion, takes corr matrices and replaces each element by
    sign(C_ij)|C_ij|^(1+e)
    """
    for df in corrmatrices:
        for i,j in itertools.product(range(len(df.index)), range(len(df.columns))):
            df.iloc[i,j] = math.copysign(1,df.iloc[i,j])* abs((df.iloc[i,j]))**(1+e)
    return corrmatrices

def getsimmatrix(supcorrmatrix):
    """
    get supressed noise matrices
    """
    emptymat= np.ndarray(shape=(len(supcorrmatrix),len(supcorrmatrix)), dtype=float)
    print("matrix shape:",emptymat.shape)
    for i, j in itertools.product(range(len(supcorrmatrix)), range(len(supcorrmatrix))):
        emptymat[i,j] = abs(supcorrmatrix[i] - supcorrmatrix[j]).mean().mean()
    
    return emptymat

def getsimtrans(simmatrix):
    """
    initialize MDS algo and fit it with our similarity matrix
    simtrans gives the new points in a reduced dimension n_components set by the MDS parameters
    """
    mds = MDS(dissimilarity = 'precomputed')
    simtrans = mds.fit_transform(simmatrix)
    
    return simtrans

def cluster_simmat(simtrans):
    km = KMeans(n_clusters=2, init='random')
    y_km = km.fit_predict(simtrans)
    
    return y_km

def intraclusterdist(simtrans, kclusters):
    """
    function to get the average (of averages) and std (average of std) of intra-cluster distances for a 
    similarity matrix that has been transformed by MDS
    """
    emptydict = {}
    avgstd = {}
    totals = []
    
    km = KMeans(n_clusters=2, init='random')  # initialize K-Means clustering algorithm
    y_km = km.fit_predict(simtrans)  # take the MDS reduced similarity matrix and use K-means clustering on it
    
    # make dictionary with cluster as key and all distances as values (array)
    for i in range(len(simtrans)):
        emptydict.setdefault(y_km[i], []).append(math.sqrt(sum((simtrans[i] - km.cluster_centers_[y_km[i]])**2)))
    # take the dictionary from above and get a new dictionary with cluster as key
    # and a value of [# of points in cluster, average distance, std]
    for key in emptydict:
        avg = sum(emptydict[key])/len(emptydict[key])
        avgstd.setdefault(key,[]).append(len(emptydict[key]))
        avgstd.setdefault(key, []).append(avg)
        # might be an error in how I calculate std here
        std = math.sqrt(sum([(dis-avg)**2 for dis in emptydict[key]])/len(emptydict[key]))
        avgstd.setdefault(key,[]).append(std)
    # now get weighted average (of average) intracluster distance and avg std (for unequal sample size)
    avgavgdis = 0 
    avgstdcounter = 0 
    for key in avgstd:
        avgavgdis += (avgstd[key][0]*avgstd[key][1])/len(simtrans)
        avgstdcounter += (avgstd[key][0]-1)*(avgstd[key][2]**2)
    avgstd = math.sqrt(avgstdcounter/(len(simtrans)-kclusters))
    
    return avgavgdis, avgstd
    