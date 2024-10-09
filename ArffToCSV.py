#!/usr/bin/env python3

import pandas as pd
from scipy.io import arff

def arff_to_csv(arff_file_path, csv_file_path):
    # Read the ARFF file
    data, meta = arff.loadarff(arff_file_path)
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Decode byte strings to normal strings
    for column in df.columns:
        if df[column].dtype == object:  # Check if the column has byte data
            df[column] = df[column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    # Save as CSV
    df.to_csv(csv_file_path, index=False)
    
    print(f"ARFF file {arff_file_path} has been successfully converted to CSV at {csv_file_path}")

# Example usage
arff_file_path = 'ML-Phishing-Link-Detector/PhiUSIIL_Phishing_URL_Dataset.arff'
csv_file_path = 'ML-Phishing-Link-Detector/PhiUSIIL_Phishing_URL_Dataset.csv'
arff_to_csv(arff_file_path, csv_file_path)