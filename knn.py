import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df=pd.read_csv("teleCust1000t.csv")

fig=plt.figure(figsize=(15,7.5))

colors = ListedColormap(['r','b','g','yellow'])
#col=np.where(df['custcat']==1,'yellow',np.where(df['custcat']==2,'r',np.where(df['custcat']==3,'g','b')))
draw=plt.scatter(df['age'],df['income'],c=df['custcat'],cmap=colors)
classes=['Basic Service','E-Service','Plus Service','Total Service']
plt.legend(handles=draw.legend_elements()[0],labels=classes,loc='best')
plt.xlabel("AGE")
plt.ylabel("INCOME")
plt.show()
#df.hist(column='income',bins=50)
#plt.show()

X=df[['region','tenure','age','marital','address','income', 'ed',
'employ', 'retire', 'gender', 'reside']].values

Y=df['custcat'].values

X=StandardScaler().fit(X).transform(X.astype(float))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=4)
#print ('Train set:', X_train.shape,  Y_train.shape)
#print ('Test set:', X_test.shape,  Y_test.shape)

k_range=10
mean_acc=np.zeros((k_range-1))
std_acc=np.zeros((k_range-1))

for i in range(1,k_range):
    neigh=KNeighborsClassifier(n_neighbors=i).fit(X_train,Y_train)
    y_hat=neigh.predict(X_test)

    mean_acc[i-1]=metrics.accuracy_score(Y_test,y_hat)
    std_acc[i-1]=np.std(y_hat==Y_test)/np.sqrt(y_hat.shape[0])

#print(mean_acc)

plt.plot(range(1,k_range),mean_acc,'g')
plt.fill_between(range(1,k_range),mean_acc-1*std_acc,mean_acc+1*std_acc,alpha=0.10)
plt.fill_between(range(1,k_range),mean_acc-3*std_acc,mean_acc+3*std_acc,alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')

plt.show()

print( "The best accuracy was with", mean_acc.max(), "with K =", mean_acc.argmax()+1) 

k=mean_acc.argmax()+1
neigh_best=KNeighborsClassifier(n_neighbors=k).fit(X_train,Y_train)

y_hat_best=neigh.predict(X_test)

print(f"Train set Accuracy : {metrics.accuracy_score(Y_train,neigh.predict(X_train))}")
print(f"Test set Accuracy : {metrics.accuracy_score(Y_test,y_hat_best)}")