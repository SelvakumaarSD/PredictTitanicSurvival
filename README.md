# PredictTitanicSurvival
This program aims to classify the samples, correlate the parameters or factors of survival and create a model that predicts the passengers’ survival.
The dataset is taken from Kaggle
https://www.kaggle.com/c/titanic

The dataset is classified into Training and Test Sets.
The Training Dataset consists of 891 samples with labels for each row showing survival of samples.
The Test Dataset contains 418 samples.

For modeling this DT prediction algorithm, the training dataset is split into train and validation sets.
Then the entropy of each characteristic (that is considered to affect passengers’ survival) is calculated using the train set.
Using this entropy information, a decision tree is derived which is validated by running the model on validation set and thereby calculating the prediction accuracy score.

Libraries used:
- Data Manipulation / Pre-Processing: numpy, pandas, missingno, re, warnings
- ML Predictive Modelling: sklearn
- Visualization: matplotlib, seaborn

Methods:
- pandas.read_csv(): Method helps read the contents of input csv file (training and test)
- pandas.to_csv(): Method helps write the results data frame to a output csv file
- missingno.matrix(): helps highlight the null valued columns from non-null ones in the data frame
- pandas.describe(): Describes the data frame passed as input to this function. Shows the data type of columns, count rows of values, etc.
- pandas.get_dummies(): helps convert categorical variable into dummy/indicator variables
- preprocessing.LabelEncoder.fit_transform(): Encode target labels with value between 0 and n_classes-1
- sklearn.model_selection.train_test_split(): Splits the into dataset into training and validation (or test) sets based on the ratio provided. Used while training a predictive model.
- Sklearn.tree.DecisionTreeClassifier(): Constructor of class Decision Tree classifier in python sklearn library. When this constructor is called, the input data frame is converted into a feature matrix (of size 891x15).
- fit(): Build a decision tree classifier from the input training set. Using the feature matrix, the entropy and information gain are calculated and stored.
- predict(): Predict the class of input validation sample or test set
- score(): Return the mean accuracy on the given test data and labels
- plot_tree(): Plots the decision tree using the class weights calculated above, so that the decision tree can be visualized
