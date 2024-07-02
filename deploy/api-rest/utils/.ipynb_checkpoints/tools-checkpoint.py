import joblib
import pandas as pd

def load_model(model_path):
    """
    Loads a machine learning model from the specified path using joblib.

    Parameters:
    - model_path (str): Path to the serialized model file (e.g., in cloud storage).

    Returns:
    - model: Loaded machine learning model object.
    """
    model = joblib.load(model_path)
    return model


def predict(model, input_data):
    """
    Performs prediction using a pre-trained machine learning model.

    Parameters:
    - model: Pre-trained machine learning model object with a `predict()` method.
    - input_data (dict): Dictionary containing input data for prediction.

    Returns:
    - int: Predicted value from the model.
    """
    input_data = pd.DataFrame([input_data])
    prediction = model.predict(input_data)
    return int(prediction[0])
