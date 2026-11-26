"""
MandiCluster - Data Loader Module
Handles loading, standardizing, validating, and filtering Agmarknet agricultural datasets.
"""

import os
import pandas as pd
import numpy as np

# Standard Agmarknet column variations mapping
COLUMN_MAPPINGS = {
    'state': 'State',
    'state_name': 'State',
    'district': 'District',
    'district_name': 'District',
    'market': 'Market',
    'market_name': 'Market',
    'mandi': 'Market',
    'commodity': 'Commodity',
    'commodity_name': 'Commodity',
    'crop': 'Commodity',
    'variety': 'Variety',
    'arrival_date': 'Arrival_Date',
    'date': 'Arrival_Date',
    'min_price': 'Min_Price',
    'minimum_price': 'Min_Price',
    'min_price_rs_quintal': 'Min_Price',
    'min_price_(rs./quintal)': 'Min_Price',
    'max_price': 'Max_Price',
    'maximum_price': 'Max_Price',
    'max_price_rs_quintal': 'Max_Price',
    'max_price_(rs./quintal)': 'Max_Price',
    'modal_price': 'Modal_Price',
    'modal_price_rs_quintal': 'Modal_Price',
    'modal_price_(rs./quintal)': 'Modal_Price'
}

REQUIRED_PRICE_COLS = ['Min_Price', 'Max_Price', 'Modal_Price']


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes incoming DataFrame column names to MandiCluster schema."""
    new_cols = {}
    for col in df.columns:
        clean_name = str(col).strip().lower().replace(" ", "_")
        if clean_name in COLUMN_MAPPINGS:
            new_cols[col] = COLUMN_MAPPINGS[clean_name]
        else:
            # Fallback title case
            new_cols[col] = col.strip()
    
    df = df.rename(columns=new_cols)
    return df


def clean_price_series(series: pd.Series) -> pd.Series:
    """Converts price strings with commas, currency symbols, or spaces into clean float."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    
    cleaned = (
        series.astype(str)
        .str.replace(r'[₹$,\s]', '', regex=True)
        .str.replace(r'[^0-9.]', '', regex=True)
    )
    return pd.to_numeric(cleaned, errors='coerce')


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads an Agmarknet CSV file, cleans numeric fields, computes agricultural metrics,
    and returns a clean, sanitized pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    # Read CSV handling various encodings
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

    # Standardize column names
    df = standardize_columns(df)

    # Ensure required categorical columns exist or fill default
    for col in ['State', 'District', 'Market', 'Commodity', 'Variety']:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()

    if 'Arrival_Date' not in df.columns:
        df['Arrival_Date'] = 'N/A'
    else:
        df['Arrival_Date'] = df['Arrival_Date'].fillna('N/A').astype(str).str.strip()

    # Clean price columns
    for price_col in REQUIRED_PRICE_COLS:
        if price_col in df.columns:
            df[price_col] = clean_price_series(df[price_col])
        else:
            df[price_col] = np.nan

    # Drop rows where all three prices are null
    df = df.dropna(subset=['Modal_Price', 'Min_Price', 'Max_Price'], how='all').copy()

    # Impute missing prices logically if partial values exist
    df['Modal_Price'] = df['Modal_Price'].fillna((df['Min_Price'] + df['Max_Price']) / 2)
    df['Min_Price'] = df['Min_Price'].fillna(df['Modal_Price'])
    df['Max_Price'] = df['Max_Price'].fillna(df['Modal_Price'])

    # Sanity check: ensure positive prices
    df = df[(df['Modal_Price'] > 0) & (df['Min_Price'] > 0) & (df['Max_Price'] > 0)].copy()

    # Reconcile Min and Max if Min > Max due to reporting typo
    needs_swap = df['Min_Price'] > df['Max_Price']
    if needs_swap.any():
        df.loc[needs_swap, ['Min_Price', 'Max_Price']] = df.loc[needs_swap, ['Max_Price', 'Min_Price']].values

    # Derived Agricultural Market Intelligence Features:
    # 1. Price Spread (intra-market price dispersion in ₹/Quintal)
    df['Price_Spread'] = df['Max_Price'] - df['Min_Price']

    # 2. Price Spread Ratio (%) - relative intraday/intramarket volatility
    df['Price_Spread_Pct'] = np.where(
        df['Modal_Price'] > 0,
        ((df['Price_Spread'] / df['Modal_Price']) * 100).round(2),
        0.0
    )

    # Reset index
    df = df.reset_index(drop=True)
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Calculates comprehensive dataset statistics and quality overview."""
    if df is None or df.empty:
        return {
            'total_records': 0,
            'total_columns': 0,
            'missing_values': {},
            'unique_states': 0,
            'unique_markets': 0,
            'unique_commodities': 0,
            'columns': [],
            'price_summary': {}
        }

    missing_dict = df.isnull().sum().to_dict()
    
    price_cols = [c for c in ['Min_Price', 'Max_Price', 'Modal_Price', 'Price_Spread', 'Price_Spread_Pct'] if c in df.columns]
    price_summary = {}
    if price_cols:
        desc = df[price_cols].describe().round(2).to_dict()
        price_summary = desc

    return {
        'total_records': int(len(df)),
        'total_columns': int(len(df.columns)),
        'missing_values': {k: int(v) for k, v in missing_dict.items()},
        'unique_states': int(df['State'].nunique()) if 'State' in df.columns else 0,
        'unique_markets': int(df['Market'].nunique()) if 'Market' in df.columns else 0,
        'unique_commodities': int(df['Commodity'].nunique()) if 'Commodity' in df.columns else 0,
        'columns': list(df.columns),
        'price_summary': price_summary
    }


def validate_csv_structure(file_path: str) -> tuple[bool, str]:
    """Validates if an uploaded CSV file contains compatible agricultural price columns."""
    try:
        try:
            sample_df = pd.read_csv(file_path, nrows=5, encoding='utf-8')
        except UnicodeDecodeError:
            sample_df = pd.read_csv(file_path, nrows=5, encoding='latin1')

        if sample_df.empty:
            return False, "Uploaded CSV file is completely empty."

        clean_sample = standardize_columns(sample_df)

        # Check for price presence
        has_prices = any(p in clean_sample.columns for p in ['Modal_Price', 'Min_Price', 'Max_Price'])
        if not has_prices:
            return False, (
                "Dataset does not contain recognizable price columns "
                "(Modal_Price, Min_Price, or Max_Price). Please ensure column headers match Agmarknet standards."
            )

        return True, "Valid agricultural dataset structure."
    except Exception as e:
        return False, f"Failed to parse CSV file: {str(e)}"
