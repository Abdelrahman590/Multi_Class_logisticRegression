import numpy as np                                      # to create arrays of numbers
import pandas as pd                                     # for loading data
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
sonar_data = pd.read_csv("Copy of sonar data.csv",header = None)
#a= sonar_data.head()          
#b= sonar_data.describe()                  #show statistical measure of the dataset like mean, stande diviation, max and min
c = sonar_data[60].value_counts()          #show the number of M (Mine) and R (Rock)
#g = sonar_data.groupby(60).mean()         #Get the mean values for all the columns to predict if the object is Mine or Rock
print (c)

#<----------------------------------------------------------------------------------------------------------------------------->
#                                         for seperating and labels data

x = sonar_data.drop(columns=60 , axis = 1)   # i store all values in x excpt last column that carry (M) and (R)
                                             #so if you need to drop column you must identify the axis = 1 
                                             # if you need to drop row you must identify the axis = 0
                                            
                                            
y = sonar_data[60]                           # i store last column in y that carry values (M) and (R)    

#<---------------------------------------------------------------------------------------------------------------------------->
#                                        for spliting our data 

#  from sklearn.model_selection import train_test_split  ----> from this function we using it to split our data
X_train,X_test,Y_train, Y_test =train_test_split(x , y , test_size = 0.1, stratify= y , random_state= 1)

#print (x.shape, X_test.shape, X_train.shape)

                                                      # X_train is a traning data
                                                      # Y_train is a label of traning data
                                                      # X_test is a testing data
                                                      # Y_test is a label of testing data
                                                    #<-------------------------------------------------------------------------->
                                                    
                                                      # test_size --> that mean we just need 10 % of data to test
                                                      # stratify  --> we need stratify to split the data depend on Rock and Mine 
                                                      # random_state  = 1 or 2 or 3 --> split the data in a particular order 
                                                      
print (X_train )
print(Y_train)

#<--------------------------------------------------------------------------------------------------------------------------------->
#                                                   Model Traning
#model traning -----> LogisticRegression 
model  = LogisticRegression()
model .fit(X_train , Y_train)

#<---------------------------------------------------------------------------------------------------------------------------------->
#                                                   Model Evaluation

X_train_prediction = model.predict(X_train)
traning_data_accuracy= accuracy_score(X_train_prediction, Y_train)
#print("Accuracy on traning data:",traning_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)
#print("Accuracy on test data:",test_data_accuracy)

#<------------------------------------------------------------------------------------------------------------------------------------>
#                                              Making a Predective System 
data_input = (0.0412,0.1135,0.0518,0.0232,0.0646,0.1124,0.1787,0.2407,0.2682,0.2058,0.56897,0.2671,0.3141,0.2904,0.3531,0.100009,0.4639,0.1859,0.4474,0.4079,0.5400,0.4786,0.4332,0.6113,0.5091,0.4606,0.7243,0.8987,0.8826,0.9201,0.8005,0.6033,0.2120,0.2866,0.4033,0.2803,0.3087,0.3550,0.2545,0.1432,0.5869,0.6431,0.5826,0.4286,0.4894,0.5777,0.4315,0.2640,0.1794,0.0772,0.0798,0.0376,0.0143,0.0272,0.0127,0.0166,0.0095,0.0225,0.0098,0.0085)
input_data_as_numpy_array = np.asarray(data_input)
input_data_reshap =input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshap)
if prediction == "R":
 print("the object is a Rock")
elif prediction =="M":
 print("the object is a mine")
else:
    print("can not define the object")