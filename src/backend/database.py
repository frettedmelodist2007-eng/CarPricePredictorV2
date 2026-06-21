import sqlite3
import os
import pandas as pd
from datetime import datetime

# Define database file path relative to this script
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
DB_PATH = os.path.join(DB_DIR, 'car_predictions.db')

def create_database():
    """
    Creates the car predictions database table if it doesn't already exist.
    """
    # Ensure the directory exists
    os.makedirs(DB_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            km_driven REAL NOT NULL,
            transmission TEXT NOT NULL,
            fuel_type TEXT NOT NULL,
            owner_type TEXT NOT NULL,
            car_age INTEGER NOT NULL,
            predicted_price REAL NOT NULL,
            confidence REAL NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(brand, model, year, km_driven, transmission, fuel_type, owner_type, car_age, predicted_price, confidence):
    """
    Saves a car price prediction entry into the database.
    """
    create_database()  # Ensure database and table exist
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO predictions (
            timestamp, brand, model, year, km_driven, transmission, fuel_type, owner_type, car_age, predicted_price, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, brand, model, int(year), float(km_driven), transmission, fuel_type, owner_type, int(car_age), float(predicted_price), float(confidence)
    ))
    
    conn.commit()
    conn.close()

def load_history(limit=50):
    """
    Loads historical prediction logs from the database as a Pandas DataFrame.
    """
    create_database()  # Ensure table exists
    
    conn = sqlite3.connect(DB_PATH)
    
    # Query logs sorted by timestamp descending
    query = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    
    conn.close()
    return df
