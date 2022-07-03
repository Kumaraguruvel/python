import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
df = pd.read_csv('iris.csv')
def func():
    pass
func()
print(df.head())
X = df.iloc[:,1:5]
Y = df.iloc[:,-1]
print(X)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y)
y_train_la = encoder.fit_transform(y_train)
y_test_la = encoder.transform(y_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train_la)
model.score(x_test,y_test_la)
from sklearn.metrics import f1_score
y_pred = model.predict(x_test)
f1_score(y_test_la,y_pred,average='weighted')
import pickle
pickle.dump(model,open('iris_mo.pkl','wb'))
model1=pickle.load(open('iris_mo.pkl','rb'))
model1.score(x_test,y_test_la)
labels=np.unique(encoder.inverse_transform(y_pred))
print(labels)
enc = np.unique(y_train_la)
print(enc)
diction= dict(zip(labels,enc))
diction
ypred1=model1.predict((x_test.iloc[1,:].values).reshape(1,-1))
print(x_test.iloc[0,:])
df1 = pd.DataFrame(y_pred,columns=['y_pred'])
df2 = pd.DataFrame(y_test_la,columns=['y_true'])
df=pd.concat([df1,df2],ignore_index=True,axis=1)
print(df)
df=df.reset_index()
a=5
a
y_pred1.reshape(1,-1)
ypred2= model.predict(x_test.iloc[0:2,:])
ypred2
(x_test.iloc[0:1,:])
ypred1
