#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier

# ignore warnings 
st.set_option('deprecation.showfileUploaderEncoding', False)

#modelling 
@st.cache(suppress_st_warning=True)
def decisionTree(tree,X_train, y_train,X_test,y_test):
    # claculate score
    score = cross_val_score(tree,X_train, y_train, cv=3, scoring='accuracy').mean() *100
    # Train the model
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    # fit model
    tree.fit(X_train, y_train)
    # make prediction
    y_pred = tree.predict(X_test)
    # calculate confuion matrix
    cm=confusion_matrix(y_test,y_pred)
    st.write('The total number of normal transactions detected is ', str(cm[0][0]))
    st.write('The total number of fraudulent transactions detected is', str(cm[1][1]))
    st.subheader('Accuracy')
    st.text("Accuracy of Decision Tree model is: ")
    st.write(score,"%")
    st.text('Confusion Matrix')
    st.write(cm)
    st.write('The confusion matrix shows', str(cm[0][0]), 'normal transactions and ', str(cm[1][1]), 
               'fraud transactions while it misclassifies ', str(cm[1][0]), 'transactions as normal transactions.' )
    st.text("Report of Decision Tree model is: ")
    #score = metrics.accuracy_score(y_test, y_pred) * 100
    cr = classification_report(y_test, y_pred)
    st.write(cr)
    st.write('The classification report describes how well the model performs when making predictions for each class. ')

    return score, cr, tree

# Training Random Forest
@st.cache(suppress_st_warning=True)
def randomForest(rf, X_train, y_train,X_test,y_test):
    # claculate score
    score = cross_val_score(rf,X_train, y_train, cv=3, scoring='accuracy').mean() *100
    # Train the model
    rf = RandomForestClassifier(random_state=42)
    # fit model
    rf.fit(X_train, y_train)
    # make prediction
    y_pred = rf.predict(X_test)
    # calculate confuion matrix
    cm=confusion_matrix(y_test,y_pred)
    st.write('The total number of normal transactions detected is ', str(cm[0][0]))
    st.write('The total number of fraudulent transactions detected is', str(cm[1][1]))
    st.subheader('Accuracy')
    st.text("Accuracy of Random Forest model is: ")
    st.write(score,"%")
    st.text('Confusion Matrix')
    st.write(cm)
    st.write('The confusion matrix shows', str(cm[0][0]), 'normal transactions and ', str(cm[1][1]), 
               'fraud transactions while it misclassifies ', str(cm[1][0]), 'transactions as normal transactions.' )
    st.text("Report of Random Forest model is: ")
    cr = classification_report(y_test, y_pred)
    st.write(cr)
    st.write('The classification report describes how well the model performs when making predictions for each class. ')

    return score, cr, rf

# Training Neural Network for Classification.
@st.cache(suppress_st_warning=True)
def neuralNet(X_train,X_test, y_train, y_test):
    # Scaling the data before feeding it to the Neural Network.
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)
    # Instantiate the Classifier and fit the model.
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    score1 = metrics.accuracy_score(y_test, y_pred)*100
    cr = classification_report(y_test, y_pred)
   
    return score1, cr, nn

# Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(knn, X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)*100
    cr = classification_report(y_test, y_pred)
    
    return score, cr, knn
 

# main
def main():
    # set heading
    st.title("A credit card fraud detection system")
    # file upload section
    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
    if file_upload is not None:
        # read the data
        df = pd.read_csv(file_upload)
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")
            # print the first 5 rows of the data
            st.write(df.head())
            # print the shape of the data
            st.subheader('Data Summary')
            st.write('Shape of the dataframe: ',df.shape)
            st.write('Data decription: \n',df.describe())
            
            # Preprocessing
            # Assign X and y; feature and target data
            X=df.drop(['Class'], axis=1)
            y=df.Class
        
            # Split the data into training and testing sets
            st.subheader('Split Data')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
            st.write('X_train: ',X_train.shape, ' y_train: ',y_train.shape)
            st.write('X_test: ',X_test.shape, ' y_test: ',y_test.shape)
       
    
    
            # ML Section
        
            choose_model = st.sidebar.selectbox("Choose the ML Model",
            ["NONE","Decision Tree", "Random Forest", "Neural Network", "K-Nearest Neighbours"])

            feat=X_train.columns.tolist()
           

            if(choose_model == "Decision Tree"):
                #Feature selection through feature importance
                model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
                @st.cache
                def feature_sort(model,X_train,y_train):
                    # fit the model
                    model.fit(X_train, y_train)
                    # get importance
                    imp = model.feature_importances_
                    return imp
    
                # Get feature importance and plot it
                st.set_option('deprecation.showPyplotGlobalUse', False)
                importance=feature_sort(model,X_train,y_train)
                feats = {} # a dict to hold feature_name: feature_importance
                for features, importances in zip(df.columns, importance):
                    feats[features] = importances #add the name/value pair 

                importances_df= pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
                importances_df.sort_values(by='Gini-importance').plot(kind='barh', rot=45)
                plt.title('Feature Importance')
                plt.xlabel('Importance')
                plt.ylabel('Features')
                st.pyplot()
    
                # get top features from the feature importance list
                feature_imp=list(zip(feat,importance))
                feature_sort=sorted(feature_imp, key = lambda x: x[1])
                n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)
                top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

                if st.sidebar.checkbox('Show selected top features'):
                    st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))
                 
                # get train and test data
                X_train_sfs=X_train[top_features]
                X_test_sfs=X_test[top_features]

                X_train_sfs_scaled=X_train_sfs
                X_test_sfs_scaled=X_test_sfs
                
                # set random seed
                np.random.seed(42)

                # set smote to handle imbalance class
                smt = SMOTE()
                
                
                st.subheader('Handling Imbalanced Class')
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                st.subheader('Model Performance')
                decisionTree(model,X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
    
        
            elif(choose_model == "Random Forest"):
                #Feature selection through feature importance
                model = RandomForestClassifier(random_state=42)
                @st.cache
                def feature_sort(model,X_train,y_train):
                    # fit the model
                    model.fit(X_train, y_train)
                    # get importance
                    imp = model.feature_importances_
                    return imp
    
                # Get feature importance and plot it
                st.set_option('deprecation.showPyplotGlobalUse', False)
                importance=feature_sort(model,X_train,y_train)
                feats = {} # a dict to hold feature_name: feature_importance
                for features, importances in zip(df.columns, importance):
                    feats[features] = importances #add the name/value pair 

                importances_df= pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
                importances_df.sort_values(by='Gini-importance').plot(kind='barh', rot=45)
                plt.title('Feature Importance')
                plt.xlabel('Importance')
                plt.ylabel('Features')
                st.pyplot()
    
                # get top features from the feature importance list
                feature_imp=list(zip(feat,importance))
                feature_sort=sorted(feature_imp, key = lambda x: x[1])
                n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)
                top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

                if st.sidebar.checkbox('Show selected top features'):
                    st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))
                 
                # get train and test data
                X_train_sfs=X_train[top_features]
                X_test_sfs=X_test[top_features]

                X_train_sfs_scaled=X_train_sfs
                X_test_sfs_scaled=X_test_sfs
                
                # set random seed
                np.random.seed(42)

                # set smote to handle imbalance class
                smt = SMOTE()
                
                
                st.subheader('Handling Imbalanced Class')
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                st.subheader('Model Performance')
                randomForest(model,X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
                
        
            elif(choose_model == "Neural Network"):
                score1, cr, nn = neuralNet(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Neural Network model is: ")
                st.write(score1,"%")
                st.text("Report of Neural Network model is: ")
                st.write(cr)
    
        
            elif(choose_model == "K-Nearest Neighbours"):
                model = KNeighborsClassifier(n_neighbors=5)
                score, cr, knn = Knn_Classifier(model,X_train, y_train, X_test, y_test)
                st.text("Accuracy of K-Nearest Neighbour model is: ")
                st.write(score,"%")
                st.text("Report of K-Nearest Neighbour model is: ")
                st.write(cr)
                

            # Visualization Section
            choose_viz = st.sidebar.selectbox("Choose the Visualization",
                ["NONE","Time Count", "Count of class","Count of Amount","Scatter plot of V11 and V14","Scatter plot of V3 and V9"])
            if choose_viz == "Time Count":
                fig = px.histogram(df['Time'], x ='Time')
                st.plotly_chart(fig)
            elif(choose_viz == "Count of class"):
                fig = px.histogram(df['Class'], x= 'Class')
                st.plotly_chart(fig)
            elif(choose_viz == "Count of Amount"):
                fig = px.histogram(df['Amount'], x= 'Amount')
                st.plotly_chart(fig)
            elif(choose_viz == "Scatter plot of V11 and V14"):
                fig = px.scatter(df , x ='V11', y ='V14')
                st.plotly_chart(fig)
            elif(choose_viz == "Scatter plot of V3 and V9"):
                fig = px.scatter(df , x ='V3', y ='V9')
                st.plotly_chart(fig)
        
        
if __name__ == "__main__":
        main()
        
