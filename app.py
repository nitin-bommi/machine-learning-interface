# importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

upload_file = st.file_uploader("Choose a CSV file to classify", type="csv")

##################################### FUNCTIONS #####################################

@st.cache
def get_dataset():

    try:
        dataset = pd.read_csv(upload_file)
    except:
        st.warning("Wrong file")

    return dataset

##################################### MAIN #####################################

if upload_file:

    dataset = get_dataset()
    st.success("Data read successfully")

    st.subheader('Display the dataset')
    if st.checkbox('Show data'):
        st.write(dataset)

    st.header("Data preprocessing")

    # feature selection

    st.subheader("Feature selection")

    features_X = list(dataset.columns)
    features_Y = list(dataset.columns)

    cols_X = st.multiselect('Select the input features(X)',
                            features_X,
                           )

    for f in cols_X:
        features_Y.remove(f)

    cols_Y = st.multiselect('Select the output features(Y)',
                            features_Y,
                           )

    X = dataset[cols_X]
    y = dataset[cols_Y]

    st.write("Input features: ", X)
    st.write("Output features: ", y)

    # missing data

    st.subheader("Handling missing data")

    missing_cols = st.multiselect('Select the features',
                                 cols_X,
                                 )

    missing_strategy = st.selectbox('Select the strategy for missing values',
                                    ['mean', 'median', 'mode', 'constant', 'drop_row'],
                                    3
                                   )

    try:

        if missing_strategy == 'mean':
            si = SimpleImputer(strategy = 'mean')
            X[missing_cols] = si.fit_transform(X[missing_cols])
        elif missing_strategy == 'median':
            si = SimpleImputer(strategy = 'median')
            X[missing_cols] = si.fit_transform(X[missing_cols])
        elif missing_strategy == 'most_frequent':
            si = SimpleImputer(strategy = 'most_frequent')
            X[missing_cols] = si.fit_transform(X[missing_cols])
        elif missing_strategy == 'constant':
            c = st.text_input("Constant: ")
            si = SimpleImputer(strategy = 'constant', fill_value = c)
            X[missing_cols] = si.fit_transform(X[missing_cols])
        elif missing_strategy == 'drop_row':
            X.dropna(inplace = True)
            X.reset_index(drop = True, inplace = True, col_fill = cols_X)

    except Exception as e:
        print(e)

    st.write("Input features after handling the missing data: ", X)

    # encoding data

    st.subheader("Encoding categorical data")

    X = pd.DataFrame(X, columns = cols_X)

    try:

        encode_cols = st.multiselect('Select the columns to perform one hot encoding', cols_X)
        X = pd.get_dummies(X, prefix = encode_cols, drop_first = True)
        cols_X = X.columns
        y = pd.get_dummies(y, drop_first = True)

    except Exception as e:
        print(e)

    st.write("Input features after encoding: ", X)

    # splitting the dataset 

    st.subheader("Splitting the data into training and test sets")

    try: 

        train_test_ratio = st.number_input('Enter the test_size', min_value = 0.1, max_value = 0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = train_test_ratio, random_state = 0)
        st.write('Rows is train set =', len(y_train))
        st.write('Rows is test set =', len(y_test))

    except Exception as e:
        print(e)
        st.error("There is some error")

    st.write("Training set: ", X_train, y_train)
    st.write("Testing set: ", X_test, y_test)

    # normalising the data

    st.subheader("Normalizing the data")

    try:

        normalize_cols = st.multiselect('Select the columns to normalize', cols_X)
        sc = StandardScaler()
        X_train[normalize_cols] = sc.fit_transform(X_train[normalize_cols])
        X_test[normalize_cols] = sc.transform(X_test[normalize_cols])

    except Exception as e:
        print(e)

    st.write("After normalizing: ", X_train, X_test)

    st.header("Model building")

    algos = ['Logistic Regression', 'K-NN', 'SVM', 'Naive Bayes', 'Decision tree', 'Random forest']
    algo_selected = st.selectbox('Select the algorithm to build the model', algos)

    if algo_selected:

        # logistic regression

        if algo_selected == algos[0]:

            C_ = st.sidebar.number_input('C', min_value=0.2, max_value=2.0)
            solver_ = st.sidebar.selectbox('solver', ['liblinear', 'lbfgs', 'saga'], 0)
            max_iter_ = st.sidebar.selectbox('max_iter', [1, 50, 100, 200, 300, 400, 500], 2)

            logistic_model = LogisticRegression(C=C_, solver=solver_, max_iter=max_iter_, random_state=0, n_jobs=-1)

            logistic_model.fit(X_train, y_train)

            logistic_pred = logistic_model.predict(X_test)
            st.write("Confusion matrix", confusion_matrix(y_test, logistic_pred))
            acc = accuracy_score(y_test, logistic_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.sidebar.checkbox('Show weights'):
                st.markdown("#### Weights ####")
                st.write("Coef: ", logistic_model.coef_)
                st.write("Intercept: ", logistic_model.intercept_)

        # k-nn

        elif algo_selected == algos[1]:

            n_neigh_ = st.sidebar.slider('n_neighbors', min_value=2, max_value=20, value=5)
            metric_ = st.sidebar.selectbox('metric', ['euclidean', 'minkowski'], 0)
            p_ = st.sidebar.selectbox('p', [1,2,3,4,5,6,7,8,9], 0)

            knn_model = KNeighborsClassifier(metric=metric_, n_neighbors=n_neigh_, p=p_, n_jobs=-1)

            knn_model.fit(X_train, y_train)

            knn_pred = knn_model.predict(X_test)
            st.write(confusion_matrix(y_test, knn_pred))
            acc = accuracy_score(y_test, knn_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

        # svm

        elif algo_selected == algos[2]:

            c_ = st.sidebar.slider('C', min_value=1, max_value=5, value=3)
            kernel_ = st.sidebar.selectbox('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], 0)
            degree_ = st.sidebar.selectbox('degree', [2, 3, 4, 5, 6, 7, 8], 0)

            svm_model = SVC(C=c_, kernel=kernel_, degree=degree_)

            svm_model.fit(X_train, y_train)

            svm_pred = svm_model.predict(X_test)
            st.write(confusion_matrix(y_test, svm_pred))
            acc = accuracy_score(y_test, svm_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.sidebar.checkbox('Show weights'):
                st.markdown("#### Model info ####")
                st.write("Number of support vectors for each class: ", svm_model.n_support_)
                if kernel_ == 'linear':
                    st.write("Coef: ", svm_model.coef_)

        # naive bayes

        elif algo_selected == algos[3]:

            naive_model = GaussianNB()

            naive_model.fit(X_train, y_train)

            naive_pred = naive_model.predict(X_test)
            st.write(confusion_matrix(y_test, naive_pred))
            acc = accuracy_score(y_test, naive_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.sidebar.checkbox('Show weights'):
                st.markdown("#### Model info ####")
                st.write("Probability of each class: ", naive_model.class_prior_)
                st.write("Variance of each feature per class: ", naive_model.sigma_)

        # decision tree

        elif algo_selected == algos[4]:

            criterion_ = st.sidebar.selectbox('criterion', ['gini', 'entropy'], 0)
            max_ = st.sidebar.selectbox('max_features', ['auto', 'log2', 'sqrt'], 0)

            decision_tree_model = DecisionTreeClassifier(criterion=criterion_, random_state=0, max_features=max_)

            decision_tree_model.fit(X_train, y_train)

            decision_tree_pred = decision_tree_model.predict(X_test)
            st.write(confusion_matrix(y_test, decision_tree_pred))
            acc = accuracy_score(y_test, decision_tree_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.sidebar.checkbox('Show weights'):
                st.markdown("#### Model info ####")
                st.write("The number of features when fit is performed: ", decision_tree_model.n_features_)
                st.write("The number of outputs when fit is performed: ", decision_tree_model.n_outputs_)

        # random forest

        elif algo_selected == algos[5]:

            n_ = st.sidebar.selectbox('n_estimators', [5,6,7,8,9,10,11,12,15], 0)
            criterion_ = st.sidebar.selectbox('criterion', ['gini', 'entropy'], 0)
            max_ = st.sidebar.selectbox('max_features', ['auto', 'log2', 'sqrt'], 0)

            random_forest_model = RandomForestClassifier(n_estimators=n_, criterion=criterion_, random_state=0 ,max_features=max_)

            random_forest_model.fit(X_train, y_train)

            random_forest_pred = random_forest_model.predict(X_test)
            st.write(confusion_matrix(y_test, random_forest_pred))
            acc = accuracy_score(y_test, random_forest_pred)

            if acc > 0.9:
                st.success('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.8:
                st.info('Accuracy = {:.4f}%'.format(acc*100))
            elif acc > 0.7:
                st.warning('Accuracy = {:.4f}%'.format(acc*100))
            else:
                st.error('Accuracy = {:.4f}%'.format(acc*100))

            if st.sidebar.checkbox('Show weights'):
                st.markdown("#### Model info ####")
                st.write("The number of features when fit is performed: ", random_forest_model.n_features_)
                st.write("The number of outputs when fit is performed: ", random_forest_model.n_outputs_)
