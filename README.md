# random_forest

Implementation of Random Forest (with bootstrapping) from scratch in python.

This project is built upon another project of mine which was implemention of DecisionTree from scratch. Fnuctions and classes that my dtree file contains:
1. LeafNode : defines a leaf of the tree. 
2. DecisionNode : defines a decision node.
3. DecisionTree : class that constructs the decision tree. In order to find the best split for tree it uses the bestsplit function (implemented in the dtree file as well; it uses gini/MSE to give the best point for the split)
4. RegressionTree621 and ClassifierTree621: built using DecisionTree. They give the ability to our implementation to work both as a regressor and classifier. Based on being Regressor or classifier fit and prdict behave differently.

implementations in rf:
1. RandomForest621: creates a forest of decision trees with bootstrapped data. This class has the ability to calculate out of bag score.
2. RandomForestRegressor621: Inherits from RandomForest. It has functions that computes score, out of bag score and predict for a regressor model.
3. RandomForestClassifier621: Inherits from RandomForest. It has functions that computes score, out of bag score and predict for a classifier model.
