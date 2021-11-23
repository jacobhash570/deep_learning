# Deep Learning 

# Background
The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With knowledge of machine learning and neural networks, features in the provided dataset will be used to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, a CSV was provided containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

 - EIN and NAME—Identification columns
 - APPLICATION_TYPE—Alphabet Soup application type
 - AFFILIATION—Affiliated sector of industry
 - CLASSIFICATION—Government organization classification
 - USE_CASE—Use case for funding
 - ORGANIZATION—Organization type
 - STATUS—Active status
 - INCOME_AMT—Income classification
 - SPECIAL_CONSIDERATIONS—Special consideration for application
 - ASK_AMT—Funding amount requested
 - IS_SUCCESSFUL—Was the money used effectively

# Instructions

## Step 1: Preprocess the data
Using Pandas and the Scikit-Learn’s StandardScaler(), the dataset was preprocessed in order to compile, train, and evaluate the neural network model later in Step 2

Using the information provided in the starter code, the following instructions were used to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in the dataset:
What variable(s) are considered the target(s) for the model?
What variable(s) are considered the feature(s) for the model?
Drop the EIN and NAME columns.
Determine the number of unique values for each column.
For those columns that have more than 10 unique values, determine the number of data points for each unique value.
Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
Use pd.get_dummies() to encode categorical variables

## Step 2: Compile, Train, and Evaluate the Model
Using TensorFlow, a neural network, or deep learning model, was designed to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Thinking about how many inputs there are before determining the number of neurons and layers in the model, compile, train, and evaluate binary classification model to calculate the model’s loss and accuracy.

Continue using the jupter notebook where the preprocessing steps from Step 1.
Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
Create the first hidden layer and choose an appropriate activation function.
If necessary, add a second hidden layer with an appropriate activation function.
Create an output layer with an appropriate activation function.
Check the structure of the model.
Compile and train the model.
Create a callback that saves the model's weights every 5 epochs.
Evaluate the model using the test data to determine the loss and accuracy.
Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.

## Step 3: Optimize the Model
Using knowledge of TensorFlow, optimize the model in order to achieve a target predictive accuracy higher than 75%. If this cannot be achieved, there will be at least three attempts.

Optimize the model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
 - Dropping more or fewer columns.
 - Creating more bins for rare occurrences in columns.
 - Increasing or decreasing the number of values for each bin.
 - Adding more neurons to a hidden layer.
 - Adding more hidden layers.
 - Using different activation functions for the hidden layers.
 - Adding or reducing the number of epochs to the training regimen.

Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
Preprocess the dataset like you did in Step 1, taking into account any modifications to optimize the model.
Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.
