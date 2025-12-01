"""
Example: Using the API
"""
import requests
import json
from datetime import datetime


def test_api():
    """Test the Predictive Maintenance API"""
    
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("Testing Predictive Maintenance API")
    print("=" * 60)
    
    # 1. Health check
    print("\n1. Health Check")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    
    # 2. Predict failure
    print("\n2. Failure Prediction")
    prediction_data = {
        "equipment_id": "EQ001",
        "sensor_data": {
            "temperature": 85.5,
            "vibration": 7.2,
            "pressure": 95.3,
            "current": 12.1,
            "voltage": 228.5,
            "rpm": 2850,
            "power": 2764.85,
            "efficiency": 99.6
        }
    }
    
    response = requests.post(
        f"{base_url}/api/v1/predict",
        json=prediction_data
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Failure Probability: {result['failure_prob']:.2%}")
        print(f"   Severity: {result['severity']}")
        print(f"   Days to Failure: {result.get('days_to_failure', 'N/A')}")
    
    # 3. Get equipment health
    print("\n3. Equipment Health Status")
    response = requests.get(f"{base_url}/api/v1/health/EQ001")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    
    # 4. Ingest sensor data
    print("\n4. Ingest Sensor Data")
    ingestion_data = {
        "equipment_id": "EQ001",
        "readings": [
            {
                "sensor_id": "TEMP_01",
                "value": 75.3,
                "unit": "celsius"
            },
            {
                "sensor_id": "VIB_01",
                "value": 4.2,
                "unit": "mm/s"
            }
        ]
    }
    
    response = requests.post(
        f"{base_url}/api/v1/data/ingest",
        json=ingestion_data
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    
    # 5. Get alerts
    print("\n5. Get Active Alerts")
    response = requests.get(f"{base_url}/api/v1/alerts")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        alerts = response.json()
        print(f"   Active Alerts: {len(alerts)}")
        if alerts:
            print(f"   Response: {json.dumps(alerts[0], indent=2)}")
    
    # 6. Get maintenance schedule
    print("\n6. Maintenance Schedule")
    response = requests.get(f"{base_url}/api/v1/maintenance/schedule?days=30")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    
    print("\n" + "=" * 60)
    print("API testing completed!")
    print("=" * 60)


if __name__ == '__main__':
    print("\nMake sure the API server is running:")
    print("  python src/api/main.py")
    print("\nThen run this script to test the endpoints.\n")
    
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API server.")
        print("Please start the server first: python src/api/main.py")
