import mnist as mn
import numpy as np
import pandas as pd
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import time

# We import sklearn.
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)


## ML methods##

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


#from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
#%matplotlib inline

#We import seaborn to make nice plots.
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})




N = 10000 #Number of data we want to use in this project


data, target = mn.get_data(N)






#First use PCA to reduce data into 50 dimensions to speed up computation
pca = PCA(n_components=50)
pca.fit(data)
W = pca.components_
pre_data = np.matmul(data, W.T)



###TSNE###

digits_proj = TSNE(n_components=2, random_state=RS).fit_transform(pre_data)

mn.scatter(digits_proj, target)


##Isomap##

iso_proj = Isomap(n_components=2).fit_transform(pre_data)
scatter(iso_proj, target)

##LLE##

lle_proj = LocallyLinearEmbedding(n_components=2).fit_transform(pre_data)
scatter(lle_proj, target)


###Using PCA###
pca = PCA(n_components=2)
pca.fit(pre_data)
W = pca.components_
PCA_project = np.matmul(pre_data, W.T)

scatter(PCA_project, target)



##mds###

clf = sklearn.manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time.time()
X_mds = clf.fit_transform(pre_data)
print("Time taken to run MDS is : ", time.time() - t0)
scatter(X_mds, target)












##Choose TSNE as our final data reduction techniques!
final_data = digits_proj

X_train, X_test, y_train, y_test = train_test_split(final_data, target, random_state=RS, test_size=0.25)


X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)


##SVM##
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
svc = OneVsRestClassifier(SVC(C = 1,kernel='rbf'))
svc = OneVsOneClassifier(SVC(C = 1,kernel='rbf'))
clf = svc.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print(accuracy_score(pred_train, y_train))
print(accuracy_score(pred_test, y_test))


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

f, ax, sc, txts = scatter(digits_proj, target)
ax.contour(xx, yy, Z, c="k", linewidths=1.2)


##Naive Bayes##

clf = OneVsOneClassifier(GaussianNB())
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print(accuracy_score(pred_train, y_train))
print(accuracy_score(pred_test, y_test))


##QDA##

clf = OneVsRestClassifier(QuadraticDiscriminantAnalysis())
clf = OneVsOneClassifier(QuadraticDiscriminantAnalysis())
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print(accuracy_score(pred_train, y_train))
print(accuracy_score(pred_test, y_test))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

#f, ax, sc, txts = scatter(digits_proj, target)
f, ax, sc, txts = scatter(X_train, y_train)
ax.contour(xx, yy, Z, c="k", linewidths=3)



##KNN##
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print(accuracy_score(pred_train, y_train))
print(accuracy_score(pred_test, y_test))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

#f, ax, sc, txts = scatter(digits_proj, target)
f, ax, sc, txts = scatter(X_train, y_train)
ax.contour(xx, yy, Z, c="k", linewidths=3)

##Logistic regression##
clf = OneVsOneClassifier(LogisticRegression())
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print(accuracy_score(pred_train, y_train))
print(accuracy_score(pred_test, y_test))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

#f, ax, sc, txts = scatter(digits_proj, target)
f, ax, sc, txts = scatter(X_train, y_train)
ax.contour(xx, yy, Z, c="k", linewidths=3)


##Random Forest##
clf = RandomForestClassifier(n_estimators=10, max_depth=10)
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print(accuracy_score(pred_train, y_train))
print(accuracy_score(pred_test, y_test))





## Gradient Boosting
clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=6, subsample=1.0)
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print(accuracy_score(pred_train, y_train))
print(accuracy_score(pred_test, y_test))


## Gaussian process #
# clf = GaussianProcessClassifier(multi_class = 'one_vs_one', max_iter_predict = 10)
# clf.fit(X_train, y_train)
# pred_train = clf.predict(X_train)
# pred_test = clf.predict(X_test)
# print(accuracy_score(pred_train, y_train))
# print(accuracy_score(pred_test, y_test))





##Model Ensembling##

def get_models():
    """Generate a library of base learners."""
    # nb = OneVsOneClassifier(GaussianNB())
    # svc = OneVsOneClassifier(SVC(C = 1,kernel='rbf'))
    # knn = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=10))
    # lr = OneVsOneClassifier(LogisticRegression())
    # qda = OneVsOneClassifier(QuadraticDiscriminantAnalysis())
    # gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=6, subsample=1.0)
    # rf = RandomForestClassifier(n_estimators=10, max_depth=10)


    nb = (GaussianNB())
    svc = (SVC(C = 1,kernel='rbf', probability=True))
    knn = (KNeighborsClassifier(n_neighbors=10))
    lr = (LogisticRegression())
    qda = (QuadraticDiscriminantAnalysis())
    gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=6, subsample=1.0)
    rf = RandomForestClassifier(n_estimators=10, max_depth=10)

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'qda': qda,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              }

    return models

def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X_train, y_train)
        P.iloc[:, i] = m.predict(X_test)
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = accuracy_score(y_test, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))



models = get_models()
P = train_predict(models)
score_models(P, y_test)


from mlens.visualization import corrmat
corrmat(P.corr(), inflate=False)


P_c = P.apply(lambda pred: (pred - y_test))
P_c[P_c != 0] = 1
corrmat(P_c.corr(), inflate=False)



def predict_prob(model_list):
    P_proba = np.zeros((len(model_list), y_test.shape[0], 10))
    print("Fitting models.")
    for i, (name, m) in enumerate(model_list.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X_train, y_train)
        P_proba[i] = m.predict_proba(X_test)
        print("done")

    print("Done.\n")
    return P_proba


P_proba = predict_prob(models)
P_ave = np.mean(P_proba, axis=0)
y_average = np.argmax(P_ave, axis=1)
print(accuracy_score(y_test, y_average))







##True ensemble Model###

#step1 Define base learners
base_learners = get_models()



#step2 Define a meta learner

meta_learner = GradientBoostingClassifier(
    n_estimators=1000,
    max_features=4,
    max_depth=3,
    subsample=0.5,
    learning_rate=0.005, 
    random_state=RS
)


#Split the training set
xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
    X_train, y_train, test_size=0.5, random_state=RS)


#Trian base learners on the training set

def train_base_learners(base_learners, inp, out, verbose=True):
    """Train all base learners in the library."""
    if verbose: print("Fitting models.")
    for i, (name, m) in enumerate(base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        m.fit(inp, out)
        if verbose: print("done")




train_base_learners(base_learners, xtrain_base, ytrain_base)


def predict_base_learners(pred_base_learners, inp, verbose=True):
    """Generate a prediction matrix."""
    P = np.zeros((inp.shape[0], 10, len(pred_base_learners)))

    if verbose: print("Generating base learner predictions.")
    for i, (name, m) in enumerate(pred_base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        p = m.predict_proba(inp)
        # With two classes, need only predictions for one class
        P[:, :,i] = p
        if verbose: print("done")

    return P


P_base = predict_base_learners(base_learners, xpred_base)
P_base = P_base.reshape(P_base.shape[0], P_base.shape[1] * P_base.shape[2])
meta_learner.fit(P_base, ypred_base)



def ensemble_predict(base_learners, meta_learner, inp, verbose=True):
    """Generate predictions from the ensemble."""
    P_pred = predict_base_learners(base_learners, inp, verbose=verbose)
    P_pred = P_pred.reshape(P_pred.shape[0], P_pred.shape[1] * P_pred.shape[2])
    return P_pred, meta_learner.predict(P_pred)


P_pred, p = ensemble_predict(base_learners, meta_learner, X_test)
print("\nEnsemble accuracy score: %.3f" % accuracy_score(y_test, p))














