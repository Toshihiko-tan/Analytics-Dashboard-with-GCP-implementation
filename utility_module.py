import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_model(filepath='linear_model.pkl'):
    """Load the pre-trained model from a file."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

def prepare_input(location, year, dia02, dia03, dia04):
    """Prepare the input data for prediction."""
    input_features = [location, year, dia02, dia03, dia04]
    test_data = pd.DataFrame([input_features], 
                             columns=['LocationAbbr', 'YearStart', 'DataValue_dia02', 'DataValue_dia03', 'DataValue_dia04'])
    return test_data

def encode_features(df, encoder, train_df_path="Data/merged_df.csv"):
    """Encode categorical features using one-hot encoding."""
    # Load training data to fit the encoder
    train_df = pd.read_csv(train_df_path)
    x_train = train_df.drop(['DataValue_dia01'], axis=1)
    encoder.fit(x_train)  # Fit encoder on the training data
    return encoder.transform(df)

def predict_diabetes(model, df, encoder):
    """Make predictions using the pre-trained model and the provided DataFrame."""
    # Encode the features before making predictions
    df_encoded = encode_features(df, encoder)
    prediction = model.predict(df_encoded)
    return prediction

def get_prediction_text(n_clicks, location, year, dia02, dia03, dia04):
    """Generate a prediction response based on user input."""
    if n_clicks > 0:
        model = load_model()
        test_data = prepare_input(location, year, dia02, dia03, dia04)
        encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['LocationAbbr'])], remainder='passthrough')
        prediction = predict_diabetes(model, test_data, encoder)
        return f'Predicted DIA01: {prediction[0]}'
    return 'Enter values and press predict.'
