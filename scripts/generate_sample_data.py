"""
Generate Sample Sensor Data for Testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(
    n_samples: int = 10000,
    n_equipment: int = 5,
    failure_rate: float = 0.1,
    output_file: str = 'data/sample_sensor_data.csv'
):
    """
    Generate synthetic sensor data for testing
    
    Args:
        n_samples: Number of data samples to generate
        n_equipment: Number of different equipment items
        failure_rate: Proportion of failure cases
        output_file: Output CSV file path
    """
    data = []
    
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_samples):
        equipment_id = f"EQ{random.randint(1, n_equipment):03d}"
        timestamp = start_date + timedelta(minutes=i)
        
        # Determine if this is a failure case
        is_failure = random.random() < failure_rate
        
        # Generate sensor readings with correlation to failure
        if is_failure:
            # Higher values and more variance indicate approaching failure
            temperature = np.random.normal(85, 10)
            vibration = np.random.normal(7.5, 2)
            pressure = np.random.normal(95, 8)
            current = np.random.normal(12, 2)
            voltage = np.random.normal(225, 15)
            rpm = np.random.normal(2800, 200)
        else:
            # Normal operating conditions
            temperature = np.random.normal(65, 5)
            vibration = np.random.normal(3, 0.5)
            pressure = np.random.normal(100, 3)
            current = np.random.normal(10, 0.5)
            voltage = np.random.normal(240, 5)
            rpm = np.random.normal(3000, 50)
        
        # Additional derived metrics
        power = voltage * current
        efficiency = (power / (power + 10)) * 100
        
        data.append({
            'equipment_id': equipment_id,
            'timestamp': timestamp,
            'temperature': max(0, temperature),
            'vibration': max(0, vibration),
            'pressure': max(0, pressure),
            'current': max(0, current),
            'voltage': max(0, voltage),
            'rpm': max(0, rpm),
            'power': power,
            'efficiency': efficiency,
            'failure': 1 if is_failure else 0
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Generated {n_samples} samples for {n_equipment} equipment items")
    print(f"Failure rate: {df['failure'].mean():.2%}")
    print(f"Data saved to {output_file}")
    
    return df


if __name__ == '__main__':
    generate_sample_data()
