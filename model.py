import joblib

def load_model(model_path):
    """
    Load the trained machine learning model.

    Parameters:
        model_path (str): The file path of the saved model.

    Returns:
        object: The trained machine learning model.
    """
    model = joblib.load(model_path)
    return model


def make_predictions(model, feature_matrix):
    """
    Make predictions using the given model.

    Parameters:
        model (object): The trained machine learning model.
        feature_matrix (DataFrame or 2D array-like): The feature matrix containing the input data.

    Returns:
        numpy.ndarray: The predicted target variable.
    """
    # Use the trained model to make predictions
    predictions = model.predict(feature_matrix)

    return predictions