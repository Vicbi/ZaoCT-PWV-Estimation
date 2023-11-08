import pandas as pd
import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats    
from sklearn import preprocessing
import pickle
from sklearn.ensemble import RandomForestRegressor
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


def select_features(dataset, model_selection, prediction_variable):
    if model_selection == 'M1':
        dataset = dataset[['brSBP', 'brDBP', 'heart_rate', 'cfPWV_tang', 'crPWV_tang', prediction_variable]]
    elif model_selection == 'M2':
        dataset = dataset[['brSBP', 'brDBP', 'cfPWV_tang', 'crPWV_tang', prediction_variable]]
    elif model_selection == 'M3':
        dataset = dataset[['MAP', 'cfPWV_tang', 'crPWV_tang', prediction_variable]]
    elif model_selection == 'M4':
        dataset = dataset[['cfPWV_tang', 'crPWV_tang', prediction_variable]]
    else:
        raise ValueError("Invalid model_selection. Choose from 'M1', 'M2', 'M3', or 'M4'.")

    return dataset

def print_results(y_test, y_pred, variable_unit):
    """
    Print various regression metrics and statistics.

    Parameters:
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        None
    """
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nrmse = 100 * rmse / (np.max(y_test) - np.min(y_test))

    print('Mean Absolute Error:', np.round(mae, 2), variable_unit)
    print('Mean Squared Error:', np.round(mse, 2), variable_unit)
    print('Root Mean Squared Error:', np.round(rmse, 2), variable_unit)
    print('Normalized Root Mean Squared Error:', np.round(nrmse, 2), '%\n')

    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
    print('Correlation:', round(r_value, 2))
    print('Slope:', round(slope, 2))
    print('Intercept:', round(intercept, 2), variable_unit)
    print('r_value:', round(r_value, 2))
    print('p_value:', round(p_value, 4))

    print('Distribution of the reference data:', round(np.mean(y_test), 3), '±', round(np.std(y_test), 3), variable_unit)
    print('Distribution of the predicted data:', round(np.mean(y_pred), 3), '±', round(np.std(y_pred), 3), variable_unit)
    
def split_features_target(dataset):
    """
    Split the input dataframe into features (X) and target (y).

    Parameters:
        dataset (pd.DataFrame): The input dataframe.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    X = np.array(dataset.iloc[:, :-1])  # All columns except the last one
    y = np.array(dataset.iloc[:, -1])   # The last column

    return X, y

def scale_data(dataset):
    """
    Scale the input dataset using Min-Max scaling.

    Parameters:
        dataset (pd.DataFrame): The dataset to be scaled.

    Returns:
        pd.DataFrame: Scaled dataset.
    """

    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(dataset.values)
    scaled_dataset = pd.DataFrame(scaled_array, columns=dataset.columns)
     
    return scaled_dataset


def rescale_values(values, prediction_variable, dataset):
    """
    Rescale values based on the specified prediction_variable and dataset.

    Parameters:
        values (numpy.ndarray): Array to be rescaled.
        prediction_variable (str): The variable being predicted.
        dataset (pandas.DataFrame): The dataset containing the prediction variable.

    Returns:
        rescaled_values (numpy.ndarray): Rescaled values.
    """
    max_prediction_variable = np.max(dataset[prediction_variable])
    min_prediction_variable = np.min(dataset[prediction_variable])
    
    rescaled_values = min_prediction_variable + (max_prediction_variable - min_prediction_variable) * values
    
    return rescaled_values


def add_random_noise(data, perc_lower, perc_upper, noise_mode):
    """
    Add random noise to the input data.

    Parameters:
        data (numpy.ndarray): Input data.
        perc_lower (float): Lower percentage for noise range.
        perc_upper (float): Upper percentage for noise range.
        noise_mode (bool): True for adding noise, False otherwise.

    Returns:
        numpy.ndarray: Noisy data if noise_mode is True, else original data.
    """
    lower = (100 - perc_lower) / 10
    upper = (100 + perc_upper) / 10
    noise = 0.1 * np.random.uniform(lower, upper, size=data.shape)

    if noise_mode:
        noisy_data = data * noise
        print('Noise added.')
    else:
        noisy_data = data
        print('No noise added.')

    return noisy_data

def select_regression_model(X_train, X_test, y_train, y_test, prediction_variable, regressor, **kwargs):
    if regressor == 'RF':
        selected_max_depth = kwargs.get('selected_max_depth', 8)
        selected_n_estimators = kwargs.get('selected_n_estimators', 100)
        bootstrap_boolean = kwargs.get('bootstrap_boolean', True)
        model, y_pred = random_forest_model(X_train, X_test, y_train, selected_max_depth, selected_n_estimators, bootstrap_boolean)

    elif regressor == 'ANN':
        selected_batch_size = kwargs.get('selected_batch_size', 10)
        if prediction_variable == 'CT_sys':
            selected_epochs = kwargs.get('selected_epochs', 55)
        elif prediction_variable == 'Z_compl':
            selected_epochs = kwargs.get('selected_epochs', 103)
        
        verbose = kwargs.get('verbose', False)
        model, y_pred = artificial_neural_networks_model(X_train, X_test, y_train, selected_batch_size, selected_epochs, verbose)

    else:
        raise ValueError("Invalid regressor specified. Use 'RF' for Random Forest or 'ANN' for Artificial Neural Network.")

    return model, y_pred


def random_forest_model(X_train,X_test,y_train,selected_max_depth,selected_n_estimators,bootstrap_boolean):
    model = RandomForestRegressor(bootstrap = bootstrap_boolean,
                                  max_depth = selected_max_depth,
                                  n_estimators = selected_n_estimators)
        
    y_pred = model.fit(X_train, y_train).predict(X_test)
    
    return model,y_pred


def artificial_neural_networks_model(X_train,X_test,y_train,selected_batch_size,selected_epochs,verbose):
    num_input_features = len(X_train[1,:])
    
    model = create_ann_architecture(num_input_features)
    
    # Fitting the ANN to the Training set
    model.fit(X_train, y_train, batch_size = selected_batch_size, epochs = selected_epochs)

    y_pred = model.predict(X_test)
    
    return model, y_pred

def create_ann_architecture(num_input_features):
    """
    Create a simple Artificial Neural Network (ANN) model.

    Parameters:
        num_input_features (int): Number of input features.

    Returns:
        Sequential: Compiled ANN model.
    """
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=num_input_features))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


def permutation_importances(model, X, y, metric):
    baseline = metric(y, model.predict(X))
    imp = []
    for col in range(X.shape[1]):
        save = X[:, col].copy()
        X[:, col] = np.random.permutation(X[:, col])
        m = metric(y, model.predict(X))
        X[:, col] = save
        imp.append(baseline - m)
    return np.array(imp)


def tune_random_forest_max_depth(X, y):
    """
    Perform hyperparameter tuning for max depth of Random Forest model.

    Parameters:
        X (array-like): Input data features.
        y (array-like): Output data values.

    Returns:
        int: Optimal max depth.
    """
    param_grid = {
        'max_depth': [3, 4, 6, 8, 10]  # List of max depth values to try
    }

    model = RandomForestRegressor(bootstrap = True, 
                                  n_estimators = 100)
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    optimal_max_depth = grid_search.best_params_['max_depth']
    
    print(f"The optimal max depth is: {optimal_max_depth}")
    
    return optimal_max_depth


def tune_ann_epochs(X_train, y_train, X_val, y_val):
    """
    Perform hyperparameter tuning for the number of epochs of an ANN using early stopping.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data target values.
        X_train (array-like): Validation data features.
        y_train (array-like): Validation data target values.
        
    Returns:
        int: Optimal number of epochs.
    """

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, callbacks=[early_stopping], verbose=0)

    optimal_epochs = early_stopping.stopped_epoch
    
    print(f"The optimal number of epochs is: {optimal_epochs}")
    
    return optimal_epochs