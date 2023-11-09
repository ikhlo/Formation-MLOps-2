import pandas as pd
import joblib

from formation_indus_ds_avancee.train_and_predict import train_model

def test_train_model():
    
    df = pd.DataFrame({
        'Q': [0.2, 0.4, 0.48, 0.7, 0.4, 0.8],
        'Ba_avg': [0.25, 0.45, 0.5, 0.75, 0.45, 0.85],
    })

    train_model(df, ".")

    model = joblib.load("model.joblib")
    X_test = pd.DataFrame({
        'Q': [0.2, 0.5, 0.1, 0.9],
    })
    
    assert len(model.predict(X_test)) == len(X_test)
    assert len(model) == 1
    