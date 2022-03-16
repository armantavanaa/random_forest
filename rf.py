import numpy as np
from dtree import *


def bootstrap(X, y, n):
    idx = np.random.randint(0, n, size=n)
    return X[idx], y[idx], idx


class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.oob_idxs = []
        self.y = None

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        tree = None
        self.y = y
        for i in range(self.n_estimators):
            X_p, y_p, idx_p = bootstrap(X, y, len(y))
            idx = [i for i in range(len(X))]
            self.oob_idxs.append(list(set(idx) - set(idx_p)))
            if self.type == 'reg':
                tree = RegressionTree621(min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features)
            elif self.type == 'clf':
                tree = ClassifierTree621(min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features)
            tree.fit(X_p, y_p)
            self.trees.append(tree)

        if self.oob_score:
            self.oob_score_ = self.oob_score_comp(X, y)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []
        self.type = 'reg'

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        all_samp = np.zeros(len(X_test))
        all_pred = np.zeros(len(X_test))
        for tree in self.trees:
            leaves = [tree.root.predict(col) for col in X_test]
            n_samp = np.array([leaf.n for leaf in leaves])
            all_samp += n_samp
            pred = np.array([leaf.prediction for leaf in leaves])
            all_pred += pred * n_samp

        return all_pred / all_samp

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_hat = self.predict(X_test)
        return r2_score(y_test, y_hat)

    def oob_score_comp(self, X, y):
        oob_counts = np.zeros(len(X))
        oob_preds = np.zeros(len(X))

        for i, tree in enumerate(self.trees):
            t_oop = self.oob_idxs[i]
            X_oob = X[t_oop]
            leaves = [tree.root.predict(col) for col in X_oob]
            leafsize = np.array([leaf.n for leaf in leaves])
            pred = np.array([leaf.prediction for leaf in leaves])
            oob_preds[t_oop] += (leafsize * pred)
            oob_counts[t_oop] += leafsize

        oob_avg_preds = oob_preds[oob_counts > 0] / oob_counts[oob_counts > 0]

        return r2_score(y[oob_counts>0], oob_avg_preds)


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []
        self.type = 'clf'

    def predict(self, X_test) -> np.ndarray:
        class_counts = np.zeros((len(X_test), len(np.unique(self.y))))
        for tree in self.trees:
            leaves = [tree.root.predict(col) for col in X_test]
            pred = np.array([leaf.prediction for leaf in leaves])
            class_counts[[i for i in range(len(X_test))], pred] += 1

        return np.argmax(class_counts, axis=1)

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_hat = self.predict(X_test)
        return accuracy_score(y_test, y_hat)

    def oob_score_comp(self, X, y):
        oob_counts = np.zeros(len(X))
        oob_preds = np.zeros((len(X), len(np.unique(y))))

        for i, tree in enumerate(self.trees):
            t_oop = self.oob_idxs[i]
            X_oob = X[t_oop]
            leaves = [tree.root.predict(col) for col in X_oob]
            leafsize = np.array([leaf.n for leaf in leaves])
            pred = np.array([leaf.prediction for leaf in leaves])
            oob_preds[t_oop, pred] += leafsize
            oob_counts[t_oop] += 1

        oob_votes = np.argmax(oob_preds[oob_counts>0], axis=1)

        return accuracy_score(y[oob_counts>0], oob_votes)