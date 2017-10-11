import pandas
import numpy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import datetime

#Загрузка данных и предобработка данных
data=pandas.read_csv('E:/Anaconda/edited_cs-training.csv')
#data=pandas.read_csv('E:/Anaconda/cs-training.csv')
data=data.drop(['Unnamed: 0'], axis=1)
#Замена пропусков на медианное значение по признаку
#data=data.fillna(round(data.median()))
#Исправление столбца коэффициента задолженности
#data.loc[data.DebtRatio > 1, 'DebtRatio'] = data['DebtRatio'].median()
#Проверка на выбросы
#for c in data.columns:
#   q1=data[c].quantile(0.25)
#    q3=data[c].quantile(0.75)
#    iqr=q3-q1
#    for v in data[c]:
#        if v not in [q1-1.5*iqr, q3+1.5*iqr]:
#            v=data[c].median()
y=data['SeriousDlqin2yrs']
x=data.drop(['SeriousDlqin2yrs'], axis=1)
cv = KFold(y.size,n_folds=5,shuffle=True)

#Случайный лес
#rfscores=[]
#for n in [300]:
#    rfscores.append(numpy.mean(cross_val_score(RandomForestRegressor(n_estimators=n,random_state=241),x,y,cv=cv,scoring='roc_auc')))
#print(rfscores)
start_time = datetime.datetime.now()
print (numpy.mean(cross_val_score(RandomForestRegressor(n_estimators=300,random_state=241),x,y,cv=cv,scoring='roc_auc')))
print ('Time elapsed:', datetime.datetime.now() - start_time)

#Градиентный бустинг
#gbscores=[]
#for n in [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]:
#    gbscores.append(numpy.mean(cross_val_score(GradientBoostingClassifier(learning_rate=0.1, n_estimators=n, verbose=True, random_state=241),x,y,cv=cv,scoring='roc_auc')))
#print(gbscores)
start_time = datetime.datetime.now()
print (numpy.mean(cross_val_score(GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, verbose=True, random_state=241),x,y,cv=cv,scoring='roc_auc')))
print ('Time elapsed:', datetime.datetime.now() - start_time)

#Логическая регрессия
scaler=StandardScaler()
x_scaled= scaler.fit_transform(x)
#lrscores=[]
#for c in range(1,200,1):
#    lrscores.append(numpy.mean(cross_val_score(LogisticRegression(penalty='l2',verbose=True,C=c,random_state=241),x_scaled,y,cv=cv,scoring='roc_auc')))
#print(lrscores)
start_time = datetime.datetime.now()
print (numpy.mean(cross_val_score(LogisticRegression(penalty='l2',verbose=True,C=1,random_state=241),x_scaled,y,cv=cv,scoring='roc_auc')))
print ('Time elapsed:', datetime.datetime.now() - start_time)   

#Загрузка тестовой выборки и предобработка данных
tdata=pandas.read_csv('E:/Anaconda/edited_cs-test.csv')
tdata=pandas.read_csv('E:/Anaconda/cs-test.csv')
tdata=tdata.drop(['Unnamed: 0'], axis=1)
tdata=tdata.drop(['SeriousDlqin2yrs'], axis=1)
#Замена пропусков на медианное значение по признаку
tdata=tdata.fillna(round(tdata.median()))
#Исправление столбца коэффициента задолженности
tdata.loc[tdata.DebtRatio > 1, 'DebtRatio'] = tdata['DebtRatio'].median()
#Проверка на выбросы
#for c in tdata.columns:
#    q1=tdata[c].quantile(0.25)
#    q3=tdata[c].quantile(0.75)
#    iqr=q3-q1
#    for v in tdata[c]:
#       if v not in [q1-1.5*iqr, q3+1.5*iqr]:
#           v=tdata[c].median()
#Предсказание для тестовой выборки
gbc=GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, verbose=True, random_state=241)
gbc.fit(x,y)
testscore=gbc.predict_proba(tdata)[:, 1]
print(testscore)
print(testscore.max())
print(testscore.min())
result = pandas.DataFrame({'financial_distress_will_be': testscore}, index=tdata.index)
result.index.name = 'id'
result.to_csv('E:/Anaconda/gmsc_result.csv') 

#ROC-кривая градиентного бустинга, случайного леса и логистической регресси
plt.figure(figsize=(10, 10))
models=[]
models.append(RandomForestClassifier(n_estimators=300,random_state=241,max_depth=4, criterion='entropy'))
models.append(GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, verbose=True, random_state=241))
models.append(LogisticRegression(penalty='l2',verbose=True,C=1,random_state=241))
for model in models:
    model.fit(x,y)
    testscore=model.predict_proba(x)[:, 1] 
    fpr, tpr, thresholds = roc_curve(y, testscore)
    roc_auc = roc_auc_score(y, testscore)
    md = str(model)
    md = md[:md.find('(')]
    plt.plot(fpr, tpr, label='ROC fold %s (auc = %0.2f)' % (md, roc_auc))
    
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
  