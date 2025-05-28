#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate sample data and upload it to a database.
This script creates sample data with different frequencies (daily, weekly, monthly)
and uploads it to either Supabase or a PostgreSQL database.
"""

# General libraries
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse
import time
import dotenv
from sqlalchemy import create_engine
import requests
import json

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.forecaster_utils import generate_sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def upload_to_postgres(df, table_name):
    """
    Upload a DataFrame to a PostgreSQL database.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to upload
    table_name : str
        Name of the table to create/replace
    """
    # Load environment variables
    dotenv.load_dotenv()
    
    # Get PostgreSQL connection parameters
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    
    # Create SQLAlchemy engine
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    
    # Upload DataFrame to PostgreSQL
    logging.info(f"Uploading {len(df)} rows to PostgreSQL table '{table_name}'")
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    logging.info(f"Successfully uploaded data to PostgreSQL table '{table_name}'")

def upload_to_supabase(df, table_name):
    """
    Upload a DataFrame to a Supabase table.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to upload
    table_name : str
        Name of the table to create/replace
    """
    # Load environment variables
    dotenv.load_dotenv()
    
    # Get Supabase connection parameters
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if supabase_url == 'your-supabase-url' or supabase_key == 'your-supabase-key':
        logging.error("Supabase credentials not configured. Please update the .env file.")
        return
    
    # Convert DataFrame to JSON
    records = df.to_dict(orient='records')
    
    # First, delete existing data (if any)
    delete_url = f"{supabase_url}/rest/v1/{table_name}?select=id"
    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal'
    }
    
    try:
        # Try to delete existing data
        logging.info(f"Clearing existing data from Supabase table '{table_name}'")
        requests.delete(delete_url, headers=headers)
        
        # Upload new data
        upload_url = f"{supabase_url}/rest/v1/{table_name}"
        logging.info(f"Uploading {len(records)} rows to Supabase table '{table_name}'")
        
        # Upload in batches to avoid request size limitations
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            response = requests.post(
                upload_url,
                headers=headers,
                data=json.dumps(batch)
            )
            response.raise_for_status()
            logging.info(f"Uploaded batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}")
        
        logging.info(f"Successfully uploaded data to Supabase table '{table_name}'")
    except Exception as e:
        logging.error(f"Error uploading to Supabase: {str(e)}")

def main():
    """Main function to generate sample data and upload to database."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate sample data and upload to database')
    parser.add_argument('--db_config', type=str, default='Supabase', choices=['Supabase', 'PostgreSQL'], 
                        help='Database configuration to use (Supabase or PostgreSQL)')
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 80)
    print("SAMPLE DATA GENERATION")
    print("=" * 80)
    print(f"• Database configuration: {args.db_config}")
    
    # Start timing
    start_time = time.time()
    
    # Generate daily sample data
    print("\nGenerating daily sample data...")
    daily_data = generate_sample_data(freq='D', periods=365)
    
    # Generate weekly sample data
    print("\nGenerating weekly sample data...")
    weekly_data = generate_sample_data(freq='W', periods=156)
    
    # Generate monthly sample data
    print("\nGenerating monthly sample data...")
    monthly_data = generate_sample_data(freq='M', periods=60)
    
    # Upload data to the selected database
    if args.db_config == 'Supabase':
        upload_to_supabase(daily_data, 'sample_data_daily')
        upload_to_supabase(weekly_data, 'sample_data_weekly')
        upload_to_supabase(monthly_data, 'sample_data_monthly')
    else:  # PostgreSQL
        upload_to_postgres(daily_data, 'sample_data_daily')
        upload_to_postgres(weekly_data, 'sample_data_weekly')
        upload_to_postgres(monthly_data, 'sample_data_monthly')
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("SAMPLE DATA GENERATION SUMMARY")
    print("=" * 80)
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Daily data shape: {daily_data.shape[0]} rows × {daily_data.shape[1]} columns")
    print(f"Weekly data shape: {weekly_data.shape[0]} rows × {weekly_data.shape[1]} columns")
    print(f"Monthly data shape: {monthly_data.shape[0]} rows × {monthly_data.shape[1]} columns")
    print("=" * 80)

if __name__ == "__main__":
    main()
