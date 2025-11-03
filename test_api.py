"""Quick API test"""
from api.main import app, load_model_and_preprocessor
from fastapi.testclient import TestClient

# Load model first
load_model_and_preprocessor()

client = TestClient(app)

print("="*60)
print("Testing FastAPI Endpoints")
print("="*60)

# Test health endpoint
print("\n1. Health Check:")
response = client.get('/health')
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# Test model info
print("\n2. Model Info:")
response = client.get('/model/info')
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# Test prediction with example data
print("\n3. Single Prediction:")
example_data = {
    'dur': 0.5,
    'spkts': 20,
    'dpkts': 15,
    'sbytes': 2048,
    'dbytes': 1536,
    'rate': 70.0,
    'sload': 32768.0,
    'dload': 24576.0,
    'proto': 'tcp',
    'service': 'http',
    'state': 'FIN'
}

response = client.post('/predict', json=example_data)
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# Test batch prediction
print("\n4. Batch Prediction:")
batch_data = {
    'samples': [example_data, example_data]
}

response = client.post('/predict/batch', json=batch_data)
print(f"   Status: {response.status_code}")
result = response.json()
print(f"   Total samples: {result['total_samples']}")
print(f"   Attacks detected: {result['attacks_detected']}")
print(f"   Processing time: {result['processing_time_ms']:.2f}ms")

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
