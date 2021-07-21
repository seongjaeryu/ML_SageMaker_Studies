from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd

from sklearn import tree
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
#     parser.add_argument('--max_leaf_nodes', type=int, default=-1)
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=300)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]

    # Here we support a single hyperparameter, 'max_leaf_nodes'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
#     max_leaf_nodes = args.max_leaf_nodes
    random_state = args.random_state
    max_iter = args.max_iter

    # Now use scikit-learn's decision tree classifier to train the model.
#     clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = MLPClassifier(random_state=random_state,
                        max_iter=max_iter)
    clf = clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

# from __future__ import print_function

# import argparse
# import os
# import pandas as pd

# # sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# # from sklearn.externals import joblib
# # Import joblib package directly
# import joblib

# ## TODO: Import any additional libraries you need to define a model
# from sklearn.neural_network import MLPClassifier

# # Provided model load function
# def model_fn(model_dir):
#     """Load model from the model_dir. This is the same model that is saved
#     in the main if statement.
#     """
#     print("Loading model.")
    
#     # load using joblib
#     model = joblib.load(os.path.join(model_dir, "model.joblib"))
#     print("Done loading model.")
    
#     return model


# ## TODO: Complete the main code
# if __name__ == '__main__':
    
#     # All of the model parameters and training parameters are sent as arguments
#     # when this script is executed, during a training job
    
#     # Here we set up an argument parser to easily access the parameters
#     parser = argparse.ArgumentParser()

#     # SageMaker parameters, like the directories for training data and saving models; set automatically
#     # Do not need to change
#     parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
#     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
#     ## TODO: Add any additional arguments that you will need to pass into your model
#     parser.add_argument('--random_state', type=int, default=1)
#     parser.add_argument('--max_iter', type=int, default=300)
    
    
    
#     # args holds all passed-in arguments
#     args = parser.parse_args()

#     # Read in csv training file
#     training_dir = args.data_dir
#     train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

#     # Labels are in the first column
#     train_y = train_data.iloc[:,0]
#     train_x = train_data.iloc[:,1:]
    
    
#     ## --- Your code here --- ##
    

#     ## TODO: Define a model 
#     hidden_layer_sizes = args.hidden_layer_sizes
#     random_state = args.random_state
#     max_iter = args.max_iter
    
#     model = MLPClassifier(random_state = random_state,
#                           max_iter = max_iter)
    
    
#     ## TODO: Train the model
#     model = model.fit(train_x, train_y)
    
    
#     ## --- End of your code  --- ##
    

#     # Save the trained model
#     joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
