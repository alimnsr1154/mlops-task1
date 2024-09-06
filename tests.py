import pytest
from model import train_model
from app import app

# Test for model.py
def test_train_model():
    # Ensure the model is trained and returns a RandomForestClassifier
    model = train_model()
    assert model is not None
    assert hasattr(model, 'predict')  # Check if the model has a predict method

# Test for Flask app
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    # Test the /predict endpoint
    response = client.post('/predict', json={'input': [5.1, 3.5, 1.4, 0.2]})
    
    # Check if the response is 200 OK
    assert response.status_code == 200
    
    # Check if the response contains a prediction
    data = response.get_json()
    assert 'prediction' in data
    assert isinstance(data['prediction'], list)

# Placeholder test
def test_placeholder():
    assert True

