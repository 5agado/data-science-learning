from sklearn.metrics import accuracy_score, log_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import svm
from sklearn import gaussian_process
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn import model_selection

import numpy as np
import time

def get_classifiers():
    clfs = {}
        #clfs['bag'] = {
        #    'clf': ensemble.BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,
        #                                      max_features=0.5), 'name': "BaggingClassifier"}
        # clfs['mlp'] = {'clf': neural_network.MLPClassifier(hidden_layer_sizes=(100,100,100), alpha=1e-5, solver='lbfgs', max_iter=500), 'name': 'MultilayerPerceptron'}

    clfs['logreg'] = {'clf': linear_model.LogisticRegression(),
                      'params': {'C': [(2**x) for x in np.arange(-5, 15, step=3)]}}
    clfs['sgd'] = {'clf': linear_model.SGDClassifier(),
                   'params': {'loss': ['perceptron'], 'alpha': 10 ** np.random.uniform(-6, 1)}}


    clfs['knc'] = {'clf':neighbors.KNeighborsClassifier(), 'params': {'n_neighbors':np.arange(3, 15)}}
    clfs['rfc'] = {'clf':ensemble.RandomForestClassifier(), 'params':{'n_estimators':np.arange(64, 1024, step=64)}}
    clfs['svc'] = {'clf': svm.SVC(), 'params': {'kernel':['linear', 'sigmoid', 'poly', 'rbf'], 'gamma':np.linspace(0.0,2.0,num=21),'C': np.linspace(0.5,1.5,num=11)}}
    clfs['abc'] = {'clf': ensemble.AdaBoostClassifier(), 'params': {'n_estimators': np.arange(64, 1024, step=64)}}
    clfs['gbc'] = {'clf': ensemble.GradientBoostingClassifier(), 'params': {'n_estimators': np.arange(64, 1024, step=64)}}

    clfs['gauss_class'] = {'clf': gaussian_process.GaussianProcessClassifier(), 'params': {}}
    clfs['gauss_nb'] = {'clf': naive_bayes.GaussianNB(), 'params': {}}

    #LinearDiscriminantAnalysis(),
    #QuadraticDiscriminantAnalysis()

    return clfs

def test_classifier(clf, X_train, y_train, X_test, y_test):
    # Train model
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    # Evaluate quality of model on test data
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    pred_probs = clf.predict_proba(X_test)
    loss = log_loss(y_test, pred_probs)

    return end - start, accuracy, loss

def test_classifiers(clfs, data, labels):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels,
                                                        test_size=0.33, random_state=42)

    for clf in clfs:
        time, accuracy, loss = test_classifier(clf, X_train, y_train, X_test,
                                                                    y_test)
        print("=" * 30)
        print(clf.__class__.__name__)
        print("Time: {:.4f}".format(time))
        print("Accuracy: {:.4%}".format(accuracy))
        print("Log Loss: {}".format(loss))

def grid_search(clfs, X_train, y_train):
    for key, clf in clfs.items():
        print(key)
        grid_clf = model_selection.GridSearchCV(clf['clf'], clf['params'])
        grid_clf.fit(X_train, y_train)
        clfs[key]['best_params'] = grid_clf.best_params_
        clfs[key]['max_test_score'] = max(grid_clf.cv_results_['mean_test_score'])


#imp = clf.feature_importances_
#names = X_train.columns
#res = pd.DataFrame({'name':names, 'importance':imp})
#res.sort_values(by=['importance'], ascending=False)[:31]