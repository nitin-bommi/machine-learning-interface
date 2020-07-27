# importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
import graphviz
import pickle

hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

# setting the title and some intro
st.title(':sparkles: Machine Learning is fun :sparkles:')

note="""
Welcome to this **Interactive Model Application** :sunglasses:.

Tired of modelling and evaluating a model?
Wanted to compare different models, but lazy?
This is for you!!!

Go ahead...
"""

def load_model():
    st.write('_To load the model, use_')
    st.code('pickle.loads(saved_model)', language='python')

st.markdown(note)

# uploading a file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:

    try:
        dataset = pd.read_csv(uploaded_file)
    except:
        st.warning('Wrong file')

    st.header('Data preprocessin')
    st.subheader('Display the dataset')
    if st.checkbox('Show data'):
        st.write(dataset)

    # selecting the columns
    st.info('Select the columns in order')

    cols = st.multiselect('Select the columns',
                            list(dataset.columns),
                            [dataset.columns[0], dataset.columns[-1]]
                        )

    st.write('You selected ', cols)

    try:
        X = dataset[cols[:-1]]
        y = dataset[cols[-1]]
    except:
        st.error("Error while reading columns")

    # handling missing data
    st.subheader('Handling missing values')
    missing_strategy = st.selectbox('Select the strategy for missing values',
                                    ['mean', 'median', 'most_frequent', 'constant'],
                                    3
                                    )

    col = list(X.columns)
    try:
        if(missing_strategy == 'constant'):
            si = SimpleImputer(missing_values=np.nan, strategy=missing_strategy, fill_value=0)
        else:
            si = SimpleImputer(missing_values=np.nan, strategy=missing_strategy)
        X = si.fit_transform(X)
        st.success('Removed nan values... Go ahead.')
    except:
        st.warning('Cannot apply the missing strategy... Try another one.')
    X = pd.DataFrame(X, columns = col)

    # one hot encoding
    try:
        l = st.multiselect('Select the columns to perform one hot encoding', cols, [cols[0]])
        sample = X
        for i in l:
            cols_before_encoding = sample.shape[1]
            col = list(sample.columns)
            ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            sample = ct.fit_transform(sample)
            cols_after_encoding = sample.shape[1]
            cols_added = cols_after_encoding - cols_before_encoding + 1
            col.remove(i)
            for n in range(cols_added,0,-1):
                s = i+str(n)
                col = [s] + col
            sample = pd.DataFrame(sample, columns=col)
        X = sample
        type(y)
        y = LabelEncoder().fit_transform(y)
    except:
        st.error("There is some error")

    # splitting the data
    st.subheader('Splitting the dataset into training and test sets')
    try:
        train_test_ratio = st.number_input('Enter the test_size', min_value=0.1, max_value=0.3)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=train_test_ratio, random_state=0)
        st.write('Rows is train set =', len(y_train))
        st.write('Rows is test set =', len(y_test))
    except:
        st.error("There is some error")

    # normalizing the data
    st.subheader('Normalizing the data')
    try:
        if st.checkbox('Normalize'):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    except:
        st.error("There is some error")

    st.header('Model')
    st.sidebar.header('Build the model')

    # choosing the model and params
    model_name = st.selectbox('Select the model',
                            ['Linear Regression','Logistic Regression', 'K-NN', 'SVM', 'Naive Bayes', 'Decision tree', 'Random forest'],
                            )

    st.subheader(model_name)
    if model_name == 'Linear Regression':
        copy_x =  st.sidebar.selectbox('copy_X',['true','false'])
        intercept =  st.sidebar.selectbox('fit_intercept',['true','false'])
        n_jobs = st.sidebar.number_input('n_jobs',-1)
        linearr_model = LinearRegression(copy_X=copy_x,fit_intercept=intercept,n_jobs=n_jobs)
        #print(linearr_model.get_params().keys())
        try:
            linearr_model.fit(X_train,y_train)
        #    linear_model.set_params(C=c_param, solver=solver_param, max_iter=max_iter_param, random_state=43)
        #    acc = linearr_model.score(X_test,y_test);
            y_pred = linearr_model.predict(X_test)
            e1 = metrics.mean_absolute_error(y_test,y_pred)
            acc1 =100-e1
            e2 = metrics.mean_squared_error(y_test,y_pred)
            acc2 = 100-e2
            e3 = np.sqrt(e2)
            acc3 =100-e3
            accu = st.selectbox('select the accuracy type',['mean_absolute_error','mean_squared_error'
            ,'square_mean_absolute_error'])
            if accu == 'mean_absolute_error':
                acc = acc1
            elif accu == 'mean_absolute_error':
                acc=acc2
            else:
                acc=acc3
            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc))

            if st.button('Save model'):
                saved_model = pickle.dumps(linearr_model)
                st.success('Model saved successfully')
                load_model()
                st.balloons()
        except :
            st.error('There is some error')
    elif model_name == 'Logistic Regression':

        c_param = st.sidebar.number_input('C', min_value=0.2, max_value=2.0)
        solver_param = st.sidebar.selectbox('solver', ['liblinear', 'lbfgs', 'saga'], 0)
        max_iter_param = st.sidebar.selectbox('max_iter', [50, 100, 200, 300, 400, 500], 0)

        logistic_model = LogisticRegression(C=c_param, solver=solver_param, max_iter=max_iter_param, random_state=43)
        try:
            logistic_model.fit(X_train,y_train)
            logistic_pred = logistic_model.predict(X_test)
            st.write(confusion_matrix(y_test,logistic_pred))
            acc = accuracy_score(y_test,logistic_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.button('Save model'):
                saved_model = pickle.dumps(logistic_model)
                st.success('Model saved successfully')
                load_model()
                st.balloons()
        except:
            st.error('There is some error')
    elif model_name == 'K-NN':

        n_neigh_param = st.sidebar.slider('n_neighbors', min_value=2, max_value=20, value=5)
        metric_param = st.sidebar.selectbox('metric', ['euclidean', 'minkowski'], 0)
        p_param = st.sidebar.selectbox('p', [1,2,3,4,5,6,7,8,9], 0)

        knn_model = KNeighborsClassifier(metric=metric_param, n_neighbors=n_neigh_param, p=p_param)
        try:
            knn_model.fit(X_train,y_train)

            knn_pred = knn_model.predict(X_test)
            st.write(confusion_matrix(y_test,knn_pred))
            acc = accuracy_score(y_test,knn_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.button('Save model'):
                saved_model = pickle.dumps(knn_model)
                st.success('Model saved successfully')
                load_model()
                st.balloons()
        except:
            st.error('There is some error')

    elif model_name == 'SVM':

        c_param = st.sidebar.slider('C', min_value=1, max_value=5, value=3)
        kernel_param = st.sidebar.selectbox('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], 0)
        degree_param = st.sidebar.selectbox('degree', [2, 3, 4, 5, 6, 7, 8], 0)

        svm_model = SVC(C=c_param, kernel=kernel_param, degree=degree_param)
        try:
            svm_model.fit(X_train,y_train)

            svm_pred = svm_model.predict(X_test)
            st.write(confusion_matrix(y_test,svm_pred))
            acc = accuracy_score(y_test,svm_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.button('Save model'):
                saved_model = pickle.dumps(decision_tree_model)
                st.success('Model saved successfully')
                load_model()
                st.balloons()
        except:
            st.error('There is some error')

    elif model_name == 'Naive Bayes':

        naive_model = GaussianNB()
        try:
            naive_model.fit(X_train,y_train)

            naive_pred = naive_model.predict(X_test)
            st.write(confusion_matrix(y_test,naive_pred))
            acc = accuracy_score(y_test,naive_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.button('Save model'):
                saved_model = pickle.dumps(naive_model)
                st.success('Model saved successfully')
                load_model()
                st.balloons()
        except:
            st.error('There is some error')

    elif model_name == 'Decision tree':

        criterion_param = st.sidebar.selectbox('criterion', ['gini', 'entropy'], 0)
        max_param = st.sidebar.selectbox('max_features', ['auto', 'log2', 'sqrt'], 0)

        decision_tree_model = DecisionTreeClassifier(criterion=criterion_param,random_state=0,max_features=max_param)
        try:
            decision_tree_model.fit(X_train,y_train)

            decision_tree_pred = decision_tree_model.predict(X_test)
            st.write(confusion_matrix(y_test,decision_tree_pred))
            acc = accuracy_score(y_test,decision_tree_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.button('Save model'):
                saved_model = pickle.dumps(decision_tree_model)
                st.success('Model saved successfully')
                load_model()
                st.balloons()
        except:
            st.error('There is some error')

        if st.sidebar.checkbox('Save tree'):
            dot_data = tree.export_graphviz(decision_tree_model, out_file=None)
            graph = graphviz.Source(dot_data)
            graph.render("visual_tree")
            st.sidebar.success("Saved successfully...")

    elif model_name == 'Random forest':

        n_param = st.sidebar.selectbox('n_estimators', [5,6,7,8,9,10,11,12,15], 0)
        criterion_param = st.sidebar.selectbox('criterion', ['gini', 'entropy'], 0)
        max_param = st.sidebar.selectbox('max_features', ['auto', 'log2', 'sqrt'], 0)

        random_forest_model = RandomForestClassifier(n_estimators=n_param,criterion=criterion_param,random_state=0,max_features=max_param)
        try:
            random_forest_model.fit(X_train,y_train)

            random_forest_pred = random_forest_model.predict(X_test)
            st.write(confusion_matrix(y_test,random_forest_pred))
            acc = accuracy_score(y_test,random_forest_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.button('Save model'):
                saved_model = pickle.dumps(random_forest_model)
                st.success('Model saved successfully')
                load_model()
                st.balloons()
        except:
            st.error('There is some error')
