import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    
    
)

Border="-"*30

####################################################################
#Step 1 : Load the dataset
####################################################################


print(Border)
print("Step 1 : Load the dataset")
print(Border)

DataPath="iris.csv"

df =pd.read_csv(DataPath)

print("Dataset loaded sucessfully")
print("Initial entries from dataset are :")
print(df.head())


####################################################################
#Step 2 : Data Analysis (EDA)
####################################################################


print(Border)
print("Step 2 : Data Analysis (EDA)")
print(Border)

print("Shape of dataset: ",df.shape)#number of row and colum
print("Colum names :",list(df.columns))
print("Missing  values per colum :")
print(df.isnull().sum())#canonical function callyeka fun ch out put dusryach ch input
print("Class Distribution (species count)")
print(df["species"].value_counts())

print("Statistical report of dataset")
print(df.describe())


####################################################################
#Step 3 : Decide Independent and dependent variables
####################################################################

print(Border)
print("Step 3 : Decide Independent and dependent variables")
print(Border)

# X=independent variables/ features
# Y=dependent variables/labels

feature_cols=[
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
    ]

X=df[feature_cols]
Y=df["species"]


print("X shape : ",X.shape)
print("Y shape :",Y.shape)


####################################################################
#Step 4 : Visualisation of Dataset
####################################################################

print(Border)
print("Step 4 : Visualisation of Dataset")
print(Border)

#creates scatter plat
plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp=df[df["species"]==sp]
    plt.scatter(temp["petal length (cm)"], temp["petal width (cm)"], label=sp)
    
plt.title("Marvellous Iris case study")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid()
plt.show()
    
####################################################################
#Step 5 : Split the  Dataset for training and testing
####################################################################

print(Border)
print("Step 5 : Split the  Dataset for training and testing")
print(Border)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5, random_state=42)
print("Data set slipting activity done")

print("X : ",X.shape)  #(150,4)
print("Y : ",Y.shape)  #(150,)

print("X_train : ",X_train.shape) #(75,4)
print("X_test : ",X_test.shape)   #(75,4)

print("Y_train : ",Y_train.shape) #(75)
print("Y_test : ",Y_test.shape)   #(75)



####################################################################
#Step 6 : Build the model
####################################################################

print(Border)
print("Step 5 : Build the model")
print(Border)


model=DecisionTreeClassifier(max_depth=5)

print("Model gets created sucessfully!")


####################################################################
#Step 7 : Train the model
####################################################################

print(Border)
print("Step 7 : Train the model")
print(Border)

model.fit(X_train,Y_train)
print("Model trained sucessfully")


####################################################################
#Step 8 : Evalute/Test the model
####################################################################

print(Border)
print("Step 8 : Evalute/Test the model")
print(Border)

Y_pred=model.predict(X_test)
print("Model testing  done")

print("Expected answer")
print(Y_test)

print("Predicted answer")
print(Y_pred)


####################################################################
#Step 9 : Evaluate the model performance
####################################################################

print(Border)
print("Step 9 : Evaluate the model performance")
print(Border)

accuracy=accuracy_score(Y_test,Y_pred)
print("Accuracy of model is :",accuracy*100)

# print("Actual values are :")
# print(list(Y_test))
# print("Predicted value are :")
# print(Y_pred)

print("Confusion matrix")
cm=confusion_matrix(Y_test,Y_pred)
print(cm)

print("Classification Report")
print(classification_report(Y_test,Y_pred))
