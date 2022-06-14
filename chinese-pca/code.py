import logging
import gensim
from gensim.models import word2vec
from gensim.models import word2vec, KeyedVectors,Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from time import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

model = KeyedVectors.load_word2vec_format('word2vec0608.txt')
'''
#单质氧化物和硫化物的分布
X = np.array(model['Cu','Fe','Zn','Ti','CuO','FeO','ZnO','TiO2','MoS2','CuS','ZnS','PbS'])#单质氧化物和硫化物的分布
pca = PCA(n_components=2) #选择需要降成的维度

Y = pca.fit_transform(X) #将100维词向量降维成2维

print(Y)
txt = ['Cu','Fe','Zn','Ti','CuO','FeO','ZnO','TiO2','MoS2','CuS','ZnS','PbS'] #图中每个点的标注

x = Y[:,0] #图中每个点的横坐标

y = Y[:,1] #图中每个点的纵坐标

plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(txt[i], xy = (x[i], y[i]), xytext = (x[i]+0.001, y[i]+0.001)) #使用matplotlib在图中画出每个点

plt.show()

'''
'''
#不同元素的分布（英文）
X = np.array(model['Cu','Fe','Zn','Ti','Mn','Co','Ni','Sn','O','H','Cl','Br','I','B','S','F','Li','Na','Mg','Ca','K'])#单质氧化物和硫化物的分布

pca = PCA(n_components=2) #选择需要降成的维度

Y = pca.fit_transform(X) #将100维词向量降维成2维

print(Y)
txt = ['Cu','Fe','Zn','Ti','Mn','Co','Ni','Sn','O','H','Cl','Br','I','B','S','F','Li','Na','Mg','Ca','K'] #图中每个点的标注

x = Y[:,0] #图中每个点的横坐标

y = Y[:,1] #图中每个点的纵坐标

plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(txt[i], xy = (x[i], y[i]), xytext = (x[i]+0.001, y[i]+0.001)) #使用matplotlib在图中画出每个点

plt.show()
'''
#不同元素的分布（中文）
X = np.array(model['铜','铁','锌','钛','锰','钴','镍','锡','氧','氢','氯','氯','碘','硼','硫','氟','锂','钠','镁','钙','钾'])#单质氧化物和硫化物的分布
pca = PCA(n_components=2) #选择需要降成的维度

Y = pca.fit_transform(X) #将100维词向量降维成2维

print(Y)
txt = ['Cu','Fe','Zn','Ti','Mn','Co','Ni','Sn','O','H','Cl','Br','I','B','S','F','Li','Na','Mg','Ca','K'] #图中每个点的标注

x = Y[:,0] #图中每个点的横坐标

y = Y[:,1] #图中每个点的纵坐标

plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(txt[i], xy = (x[i], y[i]), xytext = (x[i]+0.001, y[i]+0.001)) #使用matplotlib在图中画出每个点

plt.show()