import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn import decomposition
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

mninst = fetch_openml('mnist_784')
data = mninst.data.to_numpy()
dataset_images = np.reshape(data,(-1,28,28))
y= np.array(mninst.target)


def IntensityAverage(image):
    sum = 0
    for row in range(28):
        for col in range(28):
            sum = sum + image[row,col]
    avg = sum / (28*28)
    return avg
    
def Threshold(image):
    for row in range(28):
        for col in range(28):
            if image[row,col] > 125:
                image[row,col] = 255
            else:
                image[row,col] = 0
    return image
                
def Cal_black_area(image):
    area = 0
    for row in range(28):
        for col in range(28):
            if (image[row,col] == 255):
                area = area + 1
    return area

def Symmetry_X(image):
    counter = 0
    for row in range(14):
        for col in range(28):
            if (image[row,col] - image[27-row,col] == 0):
                counter = counter+ 1
    return counter/(28*14)

def Symmetry_Y(image):
    counter = 0
    for row in range(28):
        for col in range(14):
            if (image[row,col] - image[row,27-col] == 0):
                counter = counter+ 1
    return counter/(28*14)         

def Extract_features(dataset_images):
    dataset= np.zeros([70000,4])
    for i in range(70000):
        image=dataset_images[i]
        # The average intensity of all pixels in image
        dataset[i,0]=IntensityAverage(image)
        
        dataset_images[i]=Threshold(dataset_images[i])
        #The area of black axis
        
        dataset[i,1] = Cal_black_area(image)
        
        #The symmetry around the x axis
        dataset[i,2]= Symmetry_X(image)
        
        #The Symmerty around the Y axis
        dataset[i,3] = Symmetry_Y(image)
    return dataset
        
        
dataset= Extract_features(dataset_images)

# Find the Correlation between the first wo attributes

corr=pearsonr(dataset[0],dataset[1])
print("The correlation between the first wo attributes is {}".format(corr))

#That means they are highly related to each other

#Using principal component analysis (PCA), visualize (i.e., by using plots) the dataset with the extracted features

pca = decomposition.PCA(n_components=4)
PCA_Data = pca.fit_transform(dataset)

pca_dataframe = pd.DataFrame(data = PCA_Data, columns = ['PC1','PC2','PC3','PC4'])
pca_dataframe['y'] = y
target_names = {0:'zero',1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight',9:'Nine'}

sns.set()

sns.lmplot(x = 'PC1', y = 'PC2', data = pca_dataframe, hue = 'y')

plt.show()
#Scaling the Data

newDataScaled = StandardScaler().fit_transform(dataset)
#Split the data into 50%
df_dataset = pd.DataFrame(data = newDataScaled)
df_dataset['y'] = y
part_50 = df_dataset.sample(frac = 0.5)

#Randomly split the dataset (D and y) into 60% and 40% for training and testing purposes respectively
df_test = part_50.sample(frac = 0.4)
df_train = part_50.drop(df_test.index)


#Task 2: ML models

#Linear SVM (soft-margin): for the value of C, use grid-search cross-validation to obtain the best value from the following set of values [10, 5, 1, 0.5, 0.1, 0.05, 0.01,0.005, 0.001]. Use overall accuracy in the cross-validation process

X = df_train.iloc[:,[0,1,2,3]]

y = df_train.iloc[:,[4]]


# defining parameter range

c_values = {'C': [10, 5, 1, 0.5, 0.1, 0.05, 0.01,0.005, 0.001]}



svm_softmargin = svm.LinearSVC(max_iter=1000)



search = GridSearchCV(svm_softmargin, c_values, scoring='accuracy', n_jobs=1, cv=10, refit=True)

# execute search

result = search.fit(X, y.values.ravel())

 

print(result.best_params_)
best_model = result.best_estimator_

#SVM with RBF kernel: for the values of C and 𝜸, use grid-search cross-validation to obtain the best value from the following set of ranges (C: [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]; 𝛾: [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]). Use overall accuracy in the cross-validation process

X = df_train.iloc[:,[0,1,2,3]]
y = df_train.iloc[:,[4]]

# defining parameter range
params = {'C': [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
               'gamma': [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]}

svm_softmargin = svm.SVC(kernel="rbf", max_iter=1000)
search = GridSearchCV(svm_softmargin, params, scoring='accuracy', n_jobs=1, cv=10, refit=True)

# execute search
result = search.fit(X, y.values.ravel())
print(result.best_params_)
best_model = result.best_estimator_



# Task02: 1:Support vector machine (SVM) c: overall accuracy, and F-score
# Take the first two features to be able to visulize the results of SVM algorithm

X = df_train.iloc[:, [0,1,2,3]]
y = df_train.iloc[:, [4]]

y_names = {'zero 0','one 1', 'two 2', 'three 3', 'four 4','five 5','six 6','seven 7','eight 8','nine 9'}

###### soft margin SVM ######
C = 0.1 #from the first part
svm_softmargin = svm.LinearSVC(C=C,max_iter=10000)
model = svm_softmargin.fit(X, y.values.ravel())
y_pred=model.predict(df_train.iloc[:,[0,1,2,3]])
print(classification_report(df_train.iloc[:,[4]], y_pred, target_names=y_names))


###### soft margin SVM with RBF kernel ######
C = 0.05 #from the second part
gamma = 0.001
svm_softmargin_RBF = svm.SVC(kernel="rbf", gamma=gamma, C=C)
model = svm_softmargin_RBF.fit(X, y.values.ravel())
y_pred=model.predict(df_train.iloc[:,[0,1,2,3]])
print("Accuracy:",metrics.accuracy_score(df_train.iloc[:,[4]], y_pred))

print(classification_report(df_train.iloc[:,[4]], y_pred, target_names=y_names))


#K-nearest neighbor (KNN) algorithm:
#a. Use grid-search cross-validation to obtain the best value K from the following set of values [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]. Use overall accuracy in thecross-validation process

X = df_train.iloc[:,[0,1,2,3]]
y = df_train.iloc[:,[4]]

k_values = [3,5,7,9,11,13,15,17,19,21,23,25]
best_value_k=k_values[0]
bestk_score = 0
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    cv_result =cross_val_score(model,X, y.values.ravel(), cv = 10)
    if np.mean(cv_result) > bestk_score:
        bestk_score = np.mean(cv_result)
        best_value_k = k
print("The best value of k is {}".format(best_value_k))
print("The score of the best value of k is {}".format(bestk_score))

#b. Using the results from part a, use the testing set to report the final evaluation result of the KNN model; overall accuracy, and F-score as the evaluation metrics


x_test= df_test.iloc[:,[0,1,2,3]]
y_test = df_test.iloc[:,[4]]
model = KNeighborsClassifier(n_neighbors=25)
model.fit(X,y.values.ravel())
cv_result =cross_val_score(model,X, y.values.ravel(), cv = 10)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred,target_names = y_names))

#3- Naive Bayes algorithm: Fit the model by using a training dataset. Then, use the testing set to report the final evaluation result of the KNN model; overall accuracy, and F-score as the evaluation metrics

model = GaussianNB()
model.fit(X,y.values.ravel())
y_pred = model.predict(x_test)

print(classification_report(y_test,y_pred,target_names = y_names))

