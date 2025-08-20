"""
Section 174 Capitalization Analysis - TCJA Regime (Earnings)
===========================================================

This script analyzes earnings risks from over-capitalization of R&D under the Tax Cuts and Jobs Act of 2017 (TCJA),
which required non-GAAP capitalization of all R&D expenses under Section 174. This regime continues to apply to 
all foreign R&D activities even after OBBBA, while US-based R&D has returned to full expensing.

Key Features:
- Tests R&D to book-to-tax difference impact on earnings surprise ratio
- Independent Variable: rnd_to_btd (R&D expenses / Book-to-Tax Difference)
- Dependent Variable: earnings_surprise_ratio (Actual EPS / Estimated EPS)
- Prompts user for FMP API key
- Fetches financial data from the Financial Modeling Prep API
- Fetches earnings calendar data for earnings surprise ratio calculation
- Performs robust regression analysis with outlier detection
- Industry-specific benchmarks and S&P 500 comparisons
- Time-series analysis and trend identification
- Provides sector benchmarking and risk assessment
- Exports results to CSV and generates visualizations

Author: Diego Aguilar-Canabal
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
import requests
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os
import json
import time
import argparse
import concurrent.futures
import sys
from tqdm import tqdm
import matplotlib.colors as mcolors
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

# Global warning collector for critical data quality issues
CRITICAL_WARNINGS = []

# Cache file for earnings calendar data
EARNINGS_CALENDAR_CACHE_FILE = "earnings_calendar_cache.json"

def add_critical_warning(warning_msg):
    """Add a critical warning to the global collection."""
    CRITICAL_WARNINGS.append(warning_msg)

def print_critical_warnings_summary():
    """Print a summary of all critical warnings at the end of the analysis."""
    if CRITICAL_WARNINGS:
        print("\n" + "="*80)
        print("CRITICAL DATA QUALITY WARNINGS SUMMARY")
        print("="*80)
        print(f"Total critical warnings: {len(CRITICAL_WARNINGS)}")
        print()
        for i, warning in enumerate(CRITICAL_WARNINGS, 1):
            print(f"{i:2d}. {warning}")
        print("\n" + "="*80)
        print("RECOMMENDATION: Review these companies for data quality issues")
        print("to validate script output.")
        print("="*80)
CACHE_EXPIRY_DAYS = 1  # Reduced from 7 to 1 day for faster testing

def is_cache_expired(cache_file):
    # Cache is expired if file does not exist or is older than 30 days
    if not os.path.exists(cache_file):
        return True
    try:
        mtime = os.path.getmtime(cache_file)
        age_hours = (time.time() - mtime) / 3600
        if age_hours > 720:  # 30 days * 24 hours = 720 hours
            return True
        # --- ADDED: Explicitly return False if not expired ---
        return False
    except Exception:
        # If any error occurs, treat the cache as expired
        return True

# Generic Cache Utilities with Expiration 
def load_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"[CACHE ERROR] Corrupted cache file: {cache_file}. Error: {e}. Deleting file and reloading from API.")
            try:
                os.remove(cache_file)
            except OSError:
                pass  # File might already be deleted
            return {}
    return {}

def save_cache(cache, cache_file):
    """Enhanced cache saving with verification."""
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"[CACHE SAVED] Successfully saved {len(cache)} entries to {cache_file}")
    except Exception as e:
        print(f"[CACHE ERROR] Failed to save {cache_file}: {e}")

def verify_cache_updates(essential_caches, tickers):
    """
    Verify that caches have been properly updated after API calls.
    
    Args:
        essential_caches: List of cache file names to check
        tickers: List of tickers that should be in the caches
    
    Returns:
        dict: Verification results for each cache
    """
    verification_results = {}
    
    for cache_file in essential_caches:
        if not os.path.exists(cache_file):
            verification_results[cache_file] = {
                'status': 'MISSING',
                'message': f'Cache file {cache_file} does not exist'
            }
            continue
            
        try:
            cache = load_cache(cache_file)
            if not cache:
                verification_results[cache_file] = {
                    'status': 'EMPTY',
                    'message': f'Cache file {cache_file} is empty'
                }
                continue
                
            # Check if expected tickers are in cache
            missing_tickers = []
            for ticker in tickers[:10]:  # Check first 10 tickers as sample
                if ticker not in cache:
                    missing_tickers.append(ticker)
            
            if missing_tickers:
                verification_results[cache_file] = {
                    'status': 'INCOMPLETE',
                    'message': f'Missing {len(missing_tickers)} tickers: {missing_tickers[:5]}...',
                    'cache_size': len(cache),
                    'expected_tickers': len(tickers)
                }
            else:
                expired = is_cache_expired(cache_file)
                if expired:
                    verification_results[cache_file] = {
                        'status': 'EXPIRED',
                        'message': 'All expected tickers found, but CACHE IS EXPIRED',
                        'cache_size': len(cache),
                        'expected_tickers': len(tickers)
                    }
                else:
                    verification_results[cache_file] = {
                        'status': 'COMPLETE',
                        'message': 'All expected tickers found in fresh cache',
                        'cache_size': len(cache),
                        'expected_tickers': len(tickers)
                    }
                
        except Exception as e:
            verification_results[cache_file] = {
                'status': 'ERROR',
                'message': f'Error reading cache {cache_file}: {e}'
            }
    
    return verification_results

# === Utility to clear all caches ===
def clear_all_caches():
    for fname in [
        "quote_cache.json",
        "historical_price_cache.json", "income_statement_cache.json",
        "balance_sheet_cache.json", "cash_flow_cache.json", "sp500_cache.json",
        EARNINGS_CALENDAR_CACHE_FILE
    ]:
        if os.path.exists(fname):
            os.remove(fname)
            print(f"Deleted cache: {fname}")

# === Earnings Calendar API Functions ===
def fetch_earnings_calendar(ticker, api_key, cache_file=EARNINGS_CALENDAR_CACHE_FILE):
    """
    Fetch earnings calendar data for a given ticker from FMP API.
    
    Args:
        ticker (str): Stock ticker symbol
        api_key (str): Financial Modeling Prep API key
        cache_file (str): Cache file path
    
    Returns:
        list: Earnings calendar data or empty list if failed
    """
    # Check cache first
    cache = load_cache(cache_file)
    if ticker in cache and not is_cache_expired(cache_file):
        return cache[ticker]
    
    if api_key is None:
        print(f"[CACHE ONLY MODE] No earnings calendar cache found for {ticker}. Skipping API call because API key is None.")
        return []
    
    print(f"Fetching earnings calendar for {ticker} from API...")
    url = f"https://financialmodelingprep.com/api/v3/earnings-calendar/{ticker}?apikey={api_key}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                # Update cache
                cache[ticker] = data
                save_cache(cache, cache_file)
                return data
            else:
                print(f"[API WARNING] Empty or invalid earnings calendar data for {ticker}")
                return []
        else:
            print(f"[API ERROR] Earnings calendar fetch failed for {ticker}: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"[API ERROR] Exception fetching earnings calendar for {ticker}: {e}")
        return []

def fetch_earnings_calendar_batch(tickers, api_key, cache_file=EARNINGS_CALENDAR_CACHE_FILE):
    """
    Fetch earnings calendar data for multiple tickers in batches.
    
    Args:
        tickers (list): List of stock ticker symbols
        api_key (str): Financial Modeling Prep API key
        cache_file (str): Cache file path
    
    Returns:
        dict: Dictionary mapping tickers to earnings calendar data
    """
    results = {}
    
    for ticker in tqdm(tickers, desc="Fetching earnings calendar data"):
        data = fetch_earnings_calendar(ticker, api_key, cache_file)
        results[ticker] = data
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return results

def calculate_earnings_surprise_ratio(earnings_data, current_date):
    """
    Calculate the earnings surprise ratio (actual/estimated) for the next period.
    
    Args:
        earnings_data (list): Earnings calendar data for a ticker
        current_date (str): Current financial period date (YYYY-MM-DD)
    
    Returns:
        float: Earnings surprise ratio or np.nan if not available
    """
    if not earnings_data or not isinstance(earnings_data, list):
        return np.nan
    
    try:
        current_dt = pd.to_datetime(current_date)
        
        # Find the next earnings announcement after the current date
        next_earnings = None
        for earnings in earnings_data:
            if 'date' in earnings and 'estimatedEPS' in earnings and 'actualEPS' in earnings:
                earnings_date = pd.to_datetime(earnings['date'])
                if earnings_date > current_dt:
                    next_earnings = earnings
                    break
        
        if next_earnings and next_earnings['estimatedEPS'] is not None and next_earnings['actualEPS'] is not None:
            estimated_eps = float(next_earnings['estimatedEPS'])
            actual_eps = float(next_earnings['actualEPS'])
            
            # Avoid division by zero
            if estimated_eps != 0:
                return actual_eps / estimated_eps
            else:
                return np.nan
        
        return np.nan
        
    except Exception as e:
        print(f"[ERROR] Failed to calculate earnings surprise ratio: {e}")
        return np.nan

# === Fetch S&P500 Constituents and Tech Sector ===
SP500_CACHE_FILE = "sp500_cache.json"
SP500_URL = "https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"

def fetch_sp500_constituents(api_key):
    cache_expired = is_cache_expired(SP500_CACHE_FILE)
    if cache_expired and os.path.exists(SP500_CACHE_FILE):
        try:
            os.remove(SP500_CACHE_FILE)
            print(f"[CACHE] Deleted expired cache: {SP500_CACHE_FILE}")
        except Exception as e:
            print(f"[CACHE ERROR] Could not delete expired cache {SP500_CACHE_FILE}: {e}")
    cache = load_cache(SP500_CACHE_FILE)
    if cache and not cache_expired:
        return pd.DataFrame(cache)
    # If no cache or cache is expired, fetch from API
    if api_key is None:
        print("Warning: No S&P500 cache found and no API key provided. Cannot fetch S&P500 constituents.")
        return pd.DataFrame()
    url = SP500_URL.format(api_key=api_key)
    response = requests.get(url)
    if response.status_code == 200:
        constituents = response.json()
        df = pd.DataFrame(constituents)[["symbol", "sector", "subSector"]]
        df.columns = ["ticker", "sector", "sub_sector"]
        cache = df.to_dict('records')
        save_cache(cache, SP500_CACHE_FILE)
        return df
    else:
        print(f"[API ERROR] Failed to fetch S&P500 constituents: HTTP {response.status_code}")
        # Ensure no expired cache is used
        if os.path.exists(SP500_CACHE_FILE):
            try:
                os.remove(SP500_CACHE_FILE)
                print(f"[CACHE] Deleted expired cache after failed API call: {SP500_CACHE_FILE}")
            except Exception as e:
                print(f"[CACHE ERROR] Could not delete expired cache {SP500_CACHE_FILE}: {e}")
        return pd.DataFrame()

def get_sp500_tech(df):
    tech_df = df[df["sector"].str.lower().str.contains("technology")].copy()
    return tech_df



# === CLI/Menu for cache management ===
def cli_menu():
    parser = argparse.ArgumentParser(description="Financial Modeling Prep Analysis Tool")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all local API caches and exit.")
    parser.add_argument("--force-refresh", nargs="*", help="Force refresh for specific tickers (comma separated)")
    parser.add_argument("--tickers", nargs="*", help="Specific tickers to analyze (comma separated)")
    args = parser.parse_args()
    if args.clear_cache:
        clear_all_caches()
        print("All caches cleared. Exiting.")
        exit(0)
    if args.force_refresh:
        tickers = args.force_refresh
        for cache_file in [
            "quote_cache.json",
            "historical_price_cache.json", "income_statement_cache.json",
            "balance_sheet_cache.json", "cash_flow_cache.json"
        ]:
            cache = load_cache(cache_file)
            for ticker in tickers:
                if ticker in cache:
                    del cache[ticker]
            save_cache(cache, cache_file)
        print(f"Force-refreshed tickers: {tickers}")
        exit(0)
    return args

def get_api_key():
    """
    Prompt the user to enter their Financial Modeling Prep API key.
    
    Returns:
        str: The API key entered by the user
    """
    # Try to get the API key from an environment variable first
    api_key = os.environ.get("FMP_API_KEY")
    if api_key:
        print("✓ FMP API key found in environment variables.")
        return api_key

    print("="*60)
    print("FINANCIAL MODELING PREP API KEY REQUIRED")
    print("="*60)
    print("This script requires a Financial Modeling Prep API key to fetch financial data.")
    print("You can get a free API key at: https://financialmodelingprep.com/developer/docs/")
    print()
    
    # Prompt user for API key
    try:
        api_key = input("Please enter your Financial Modeling Prep API key: ").strip()
        if api_key:
            print("✓ API key entered successfully.")
            return api_key
        else:
            print("No API key provided. Cannot proceed without API key.")
            return None
    except (KeyboardInterrupt, EOFError):
        print("\nNo API key provided. Cannot proceed without API key.")
        return None

def fetch_statements(ticker, api_key):
    """
    Fetch financial statements for a given ticker symbol.
    
    This function makes API calls to get balance sheet, income statement, and cash flow data.
    It returns three pandas DataFrames containing the financial data.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'MSFT', 'AAPL')
        api_key (str): Financial Modeling Prep API key
    
    Returns:
        tuple: (balance_sheet_df, income_statement_df, cash_flow_df)
    """
    # Diagnostics: Print raw API response for the first ticker only
    if ticker == "AAPL":
        for endpoint in ["balance-sheet-statement", "income-statement", "cash-flow-statement"]:
            if api_key is None:
                print(f"[CACHE ONLY MODE] No {endpoint} cache found for {ticker}. Skipping API call because API key is None.")
                continue
            url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period=quarter&limit=20&apikey={api_key}"
            print(f"[STATEMENT DIAGNOSTIC] Fetching {endpoint} for {ticker}: {url}")
            resp = requests.get(url)
            print(f"[STATEMENT DIAGNOSTIC] Status code: {resp.status_code}")
            print(f"[STATEMENT DIAGNOSTIC] Response: {resp.text[:500]}")
    bs, bs_cache = fetch_balance_sheet(ticker, api_key)
    is_, is_cache = fetch_income_statement(ticker, api_key)
    cf, cf_cache = fetch_cash_flow(ticker, api_key)
    # Always convert to DataFrame if not already
    if not isinstance(bs, pd.DataFrame):
        bs = pd.DataFrame(bs)
    if not isinstance(is_, pd.DataFrame):
        is_ = pd.DataFrame(is_)
    if not isinstance(cf, pd.DataFrame):
        cf = pd.DataFrame(cf)
    # Diagnostics for the first ticker
    if 'tickers' in globals() and ticker == tickers[0]:
        print(f"[DEBUG] {ticker} balance sheet type: {type(bs)}, shape: {bs.shape}")
        print(f"[DEBUG] {ticker} income statement type: {type(is_)}, shape: {is_.shape}")
        print(f"[DEBUG] {ticker} cash flow type: {type(cf)}, shape: {cf.shape}")
        if bs.empty: print(f"[DEBUG] {ticker} balance sheet contents: {bs}")
        if is_.empty: print(f"[DEBUG] {ticker} income statement contents: {is_}")
        if cf.empty: print(f"[DEBUG] {ticker} cash flow contents: {cf}")
    # Defensive: if any are None, return empty DataFrames and False
    if bs is None or is_ is None or cf is None:
        return ( (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), False )
    # Diagnostics: Print DataFrame shapes after conversion
    if ticker == "AAPL":
        print(f"[STATEMENT DIAGNOSTIC] balance_sheet_df shape: {pd.DataFrame(bs).shape}")
        print(f"[STATEMENT DIAGNOSTIC] income_statement_df shape: {pd.DataFrame(is_).shape}")
        print(f"[STATEMENT DIAGNOSTIC] cash_flow_df shape: {pd.DataFrame(cf).shape}")
    loaded_from_cache = bs_cache and is_cache and cf_cache
    # Ensure bs, is_, cf are DataFrames, not None
    if bs is None:
        bs = pd.DataFrame()
    if is_ is None:
        is_ = pd.DataFrame()
    if cf is None:
        cf = pd.DataFrame()
    return ( (pd.DataFrame(bs), pd.DataFrame(is_), pd.DataFrame(cf)), loaded_from_cache )

def get_sp500_data(api_key):
    """
    Fetch S&P 500 index data for benchmarking.
    
    Args:
        api_key (str): Financial Modeling Prep API key
    
    Returns:
        dict: S&P 500 metrics
    """
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={api_key}"
        data = requests.get(url).json()
        if data and len(data) > 0:
            return {
                'price': float(data[0]['price']),
                'change': float(data[0]['change']),
                'changePercent': float(data[0]['changesPercentage'])
            }
        return None
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        return None


# === Batch API Call Utility ===
def fetch_in_batches(tickers, endpoint_url, api_key, cache_file, batch_size=100, key_field="symbol", fields=None, sleep_time=1):
    cache = load_cache(cache_file)
    updated_cache = cache.copy()
    results = []
    
    # If no API key provided, only load from cache
    if api_key is None:
        print(f"[CACHE ONLY MODE] Loading {endpoint_url.split('/')[-1]} data from cache only - no API calls")
        for ticker in tickers:
            if ticker in cache:
                row = cache[ticker].copy()
                row["ticker"] = ticker
                results.append(row)
        return pd.DataFrame(results)
    
    # Normal API fetching mode
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        uncached_batch = [t for t in batch if t not in cache]
        if uncached_batch:
            url = f"{endpoint_url}/{','.join(uncached_batch)}?apikey={api_key}"
        
            try:
                response = requests.get(url)
            
                # --- ADDED: Print raw response content for the first batch only (quotes endpoint) ---
                if endpoint_url.endswith("/quote") and i == 0:
                    print(f"[API DIAGNOSTIC] Raw response content for quotes (first batch): {response.text[:1000]}")
                # --- END ADDED ---
                if response.status_code == 200:
                    batch_data = response.json()
                    print(f"[API DIAGNOSTIC] Sample data: {str(batch_data)[:500]}")
                    for entry in batch_data:
                        ticker = entry.get(key_field)
                        if ticker:
                            if fields:
                                updated_cache[ticker] = {f: entry.get(f) for f in fields}
                    save_cache(updated_cache, cache_file)
            except Exception as e:
                print(f"[API ERROR] Exception for batch: {batch}")
                print(e)
                print("[API ERROR] Retrying once...")
                try:
                    response = requests.get(url)
                    print(f"[API DIAGNOSTIC] Retry status code: {response.status_code}")
                    if response.status_code == 200:
                        batch_data = response.json()
                        print(f"[API DIAGNOSTIC] Retry sample data: {str(batch_data)[:500]}")
                        for entry in batch_data:
                            ticker = entry.get(key_field)
                            if ticker:
                                if fields:
                                    updated_cache[ticker] = {f: entry.get(f) for f in fields}
                        save_cache(updated_cache, cache_file)
                except Exception as e2:
                    print(f"[API ERROR] Retry exception for batch: {batch}")
                    print(e2)
        # Always collect the batch (whether cached or fresh)
        for ticker in batch:
            if ticker in updated_cache:
                row = updated_cache[ticker].copy()
                row["ticker"] = ticker
                results.append(row)
    time.sleep(sleep_time)
    
    # --- ADDED: Print raw response content for quotes at the end ---
    if endpoint_url.endswith("/quote"):
        print(f"[QUOTES DIAGNOSTIC] Final quotes data shape: {len(results)} rows")
        if results:
            print(f"[QUOTES DIAGNOSTIC] Sample quotes result: {results[0] if len(results) > 0 else 'No results'}")
            if len(results) > 1:
                print(f"[QUOTES DIAGNOSTIC] Additional tickers: {[r.get('ticker', 'N/A') for r in results[1:5]]}...")
    # --- END ADDED ---
    
    return pd.DataFrame(results)

# === Example: Batch /quote fetch ===
def fetch_quotes_in_batches(tickers, api_key, cache_file="quote_cache.json"):
    """
    Fetch quote data with proper cache management and API fallback.
    """
    # Load existing cache
    cache = load_cache(cache_file)
    if cache is None:
        cache = {}
    
    # Check if cache is expired
    cache_expired = is_cache_expired(cache_file)
    
    # If no API key provided, only load from cache
    if api_key is None:
        print("[CACHE ONLY MODE] Loading quote data from cache - no API calls")
        results = []
        for ticker in tickers:
            if ticker in cache:
                results.append(cache[ticker])
        return pd.DataFrame(results)
    
    # If cache is not expired, check if we have all required tickers
    if not cache_expired:
        missing_tickers = [ticker for ticker in tickers if ticker not in cache]
        if not missing_tickers:
            print("[CACHE HIT] All quote data available in cache")
            results = []
            for ticker in tickers:
                if ticker in cache:
                    results.append(cache[ticker])
            return pd.DataFrame(results)
        else:
            print(f"[CACHE MISS] Missing {len(missing_tickers)} tickers in quote cache")
    
    # Fetch missing data from API
    print(f"[API FETCH] Fetching quote data for {len(tickers)} tickers")
    results = []
    
    # Use batch fetching for efficiency
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        url = f"https://financialmodelingprep.com/api/v3/quote/{','.join(batch)}?apikey={api_key}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                batch_data = response.json()
                if isinstance(batch_data, list):
                    for item in batch_data:
                        if 'symbol' in item:
                            ticker = item['symbol']
                            cache[ticker] = item
                            results.append(item)
                time.sleep(0.1)  # Rate limiting
            else:
                print(f"[API ERROR] Quote fetch failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"[API ERROR] Exception fetching quotes: {e}")
    
    # Save updated cache
    save_cache(cache, cache_file)
    print(f"[CACHE SAVED] Saved {len(cache)} quote entries to {cache_file}")
    
    return pd.DataFrame(results)

# === Cached fetch for /key-metrics (robust, with progress bar and diagnostics) ===
def fetch_key_metrics_one_by_one(tickers, api_key, cache_file="key_metrics_cache.json"):
    """
    Fetch key metrics data, but use profile cache as the primary source to eliminate redundancy.
    Only fetch from key-metrics endpoint if profile data is not available.
    """
    profile_cache = load_cache("profile_cache.json")
    results = {}
    failed = []
    
    # Always prefer profile cache data over separate key metrics cache
    print("[PROFILE CACHE MODE] Using profile cache as primary source for key metrics")
    for ticker in tickers:
        if ticker in profile_cache:
            profile = profile_cache[ticker]
            # Extract key metrics from profile data
            key_metrics = [{
                'roe': profile.get('returnOnEquity'),
                'payoutRatio': profile.get('payoutRatio'),
                'peRatio': profile.get('pe'),
                'priceToSalesRatio': profile.get('priceToSalesRatio'),
                'debtToEquity': profile.get('debtToEquityRatio'),
                'netIncomePerShare': profile.get('eps')
            }]
            results[ticker] = key_metrics
    # Only fetch from key-metrics endpoint if we have API key and profile data is missing
    if api_key is not None and failed:
        print(f"[FALLBACK] Fetching key metrics for {len(failed)} tickers missing from profile cache")
        for ticker in failed:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?limit=1&apikey={api_key}"
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        results[ticker] = data
                time.sleep(0.22)  # Rate limiting
            except Exception as e:
                print(f"[API ERROR] Exception fetching key metrics for {ticker}: {e}")
    
    if failed:
        print(f"[SUMMARY] Missing key metrics data for: {failed}")
    return results

def fetch_ratios_in_batches(tickers, api_key, cache_file="ratios_cache.json"):
    """
    Fetch ratios data with proper cache management and API fallback.
    """
    # Load existing cache
    cache = load_cache(cache_file)
    if cache is None:
        cache = {}
    
    # Check if cache is expired
    cache_expired = is_cache_expired(cache_file)
    
    # If no API key provided, only load from cache
    if api_key is None:
        print("[CACHE ONLY MODE] Loading ratios data from cache - no API calls")
        results = {}
        for ticker in tickers:
            if ticker in cache:
                results[ticker] = cache[ticker]
        return results
    
    # If cache is not expired, check if we have all required tickers
    if not cache_expired:
        missing_tickers = [ticker for ticker in tickers if ticker not in cache]
        if not missing_tickers:
            print("[CACHE HIT] All ratios data available in cache")
            results = {}
            for ticker in tickers:
                if ticker in cache:
                    results[ticker] = cache[ticker]
            return results
        else:
            print(f"[CACHE MISS] Missing {len(missing_tickers)} tickers in ratios cache")
    
    # Fetch missing data from API
    print(f"[API FETCH] Fetching ratios data for {len(tickers)} tickers")
    results = {}
    
    # Use individual fetching for ratios (API doesn't support batch)
    for ticker in tickers:
        if ticker in cache and not cache_expired:
            results[ticker] = cache[ticker]
            continue
            
        url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?limit=1&apikey={api_key}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    cache[ticker] = data
                    results[ticker] = data
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"[API ERROR] Exception fetching ratios for {ticker}: {e}")
    
    # Save updated cache
    save_cache(cache, cache_file)
    print(f"[CACHE SAVED] Saved {len(cache)} ratios entries to {cache_file}")
    
    return results

# === Cached fetch for /historical-price-full (single ticker) ===
def fetch_historical_price_full(ticker, api_key, cache_file="historical_price_cache.json"):
    cache = load_cache(cache_file)
    if ticker in cache:
        return cache[ticker]
    if api_key is None:
        print(f"[CACHE ONLY MODE] No historical price cache found for {ticker}. Skipping API call because API key is None.")
        return None
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        cache[ticker] = data
        save_cache(cache, cache_file)
        return data

def fetch_statement_cached(endpoint, ticker, api_key, cache_file, return_cache_status=False):
    cache = load_cache(cache_file)
    if ticker in cache:
        if return_cache_status:
            return cache[ticker], True
        return cache[ticker]
    if api_key is None:
        print(f"[CACHE ONLY MODE] No {endpoint} cache found for {ticker}. Skipping API call because API key is None.")
        if return_cache_status:
            return None, False
        return None
    print(f"Fetching {endpoint} for {ticker} from API...")
    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period=quarter&limit=8&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 429 or (response.status_code == 200 and 'Limit Reach' in response.text):
        print("\n[API ERROR] You have reached your Financial Modeling Prep API plan limit. No further data can be fetched. Please wait for your quota to reset or upgrade your plan.\n")
        sys.exit(1)
    if response.status_code == 200:
        data = response.json()
        cache[ticker] = data
        save_cache(cache, cache_file)
        if return_cache_status:
            return data, False
        return data

def fetch_statements_batch(tickers, api_key, batch_size=50):
    # Defensive: always return a dictionary
    if tickers is None:
        return {}
    results = {}
    failed_tickers = []
    total = len(tickers)
    
    # If API_KEY is None, we're using cached data only
    if api_key is None:
        with tqdm(total=total, desc="Loading from local cache") as pbar:
            for ticker in tickers:
                try:
                    result, loaded_from_cache = fetch_statements(ticker, api_key)
                    if result and any(len(df) > 0 for df in result):
                        results[ticker] = result
                    pbar.update(1)
                except Exception as e:
                    print(f"Error loading cached data for {ticker}: {e}")
                    failed_tickers.append(ticker)
                    pbar.update(1)
    else:
        # Fetch from API for each ticker
        with tqdm(total=total, desc="Fetching from API") as pbar:
            for ticker in tickers:
                try:
                    result, loaded_from_cache = fetch_statements(ticker, api_key)
                    if result and any(len(df) > 0 for df in result):
                        results[ticker] = result
                    pbar.update(1)
                except Exception as e:
                    print(f"Error fetching data from API for {ticker}: {e}")
                    failed_tickers.append(ticker)
                    pbar.update(1)
    if failed_tickers:
        print("\nWARNING: Could not retrieve data for the following tickers:")
        print(", ".join(failed_tickers))
    # At the end, ensure return value is a dict
    if results is None:
        return {}
    return results

def fetch_income_statement(ticker, api_key, cache_file="income_statement_cache.json"):
    cache_expired = is_cache_expired(cache_file)
    cache = load_cache(cache_file)  # This will return {} if file doesn't exist
    if cache and not cache_expired and ticker in cache:
        return cache[ticker], True  # True = loaded from cache
    if not api_key:
        print(f"[CACHE ONLY MODE] No API key set, and cache for {ticker} is missing or expired.")
        return [], False
    if api_key is None:
        print(f"[CACHE ONLY MODE] No income statement cache found for {ticker}. Skipping API call because API key is None.")
        return [], False
    # Fetch from API - get extra periods for rolling 4-quarter calculations and lookahead
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&limit=16&apikey={api_key}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                cache[ticker] = data
                save_cache(cache, cache_file)
                return data, False
            else:
                print(f"[API WARNING] Empty or invalid income statement data for {ticker}")
                return [], False
        else:
            print(f"[API ERROR] Income statement fetch failed for {ticker}: HTTP {response.status_code}")
            return [], False
    except Exception as e:
        print(f"[API ERROR] Exception fetching income statement for {ticker}: {e}")
        return [], False

def fetch_balance_sheet(ticker, api_key, cache_file="balance_sheet_cache.json"):
    cache_expired = is_cache_expired(cache_file)
    cache = load_cache(cache_file)  # This will return {} if file doesn't exist
    if cache and not cache_expired and ticker in cache:
        return cache[ticker], True
    if not api_key:
        print(f"[CACHE ONLY MODE] No API key set, and cache for {ticker} is missing or expired.")
        return [], False
    if api_key is None:
        print(f"[CACHE ONLY MODE] No balance sheet cache found for {ticker}. Skipping API call because API key is None.")
        return [], False
    # Fetch from API - get extra periods for rolling 4-quarter calculations and lookahead
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=quarter&limit=16&apikey={api_key}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                cache[ticker] = data
                save_cache(cache, cache_file)
                return data, False
            else:
                print(f"[API WARNING] Empty or invalid balance sheet data for {ticker}")
                return [], False
        else:
            print(f"[API ERROR] Balance sheet fetch failed for {ticker}: HTTP {response.status_code}")
            return [], False
    except Exception as e:
        print(f"[API ERROR] Exception fetching balance sheet for {ticker}: {e}")
        return [], False

def fetch_cash_flow(ticker, api_key, cache_file="cash_flow_cache.json"):
    cache_expired = is_cache_expired(cache_file)
    cache = load_cache(cache_file)  # This will return {} if file doesn't exist
    if cache and not cache_expired and ticker in cache:
        return cache[ticker], True
    if not api_key:
        print(f"[CACHE ONLY MODE] No API key set, and cache for {ticker} is missing or expired.")
        return [], False
    if api_key is None:
        print(f"[CACHE ONLY MODE] No cash flow cache found for {ticker}. Skipping API call because API key is None.")
        return [], False
    # Fetch from API - get extra periods for rolling 4-quarter calculations and lookahead
    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=quarter&limit=16&apikey={api_key}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                cache[ticker] = data
                save_cache(cache, cache_file)
                return data, False
            else:
                print(f"[API WARNING] Empty or invalid cash flow data for {ticker}")
                return [], False
        else:
            print(f"[API ERROR] Cash flow fetch failed for {ticker}: HTTP {response.status_code}")
            return [], False
    except Exception as e:
        print(f"[API ERROR] Exception fetching cash flow for {ticker}: {e}")
        return [], False

# Main Analysis
# =============
def main():
    
    # Helper function for DataFrame validation
    def create_robust_mask(df, X, y, mask_robust, mask_abs):
        """
        Create a robust mask for filtering DataFrames using pandas operations.
        
        Args:
            df: Original DataFrame
            X: Feature matrix
            y: Target vector
            mask_robust: Robust outlier mask
            mask_abs: Absolute value mask
            
        Returns:
            Boolean mask for filtering
        """
        # Use pandas operations for efficient validation
        invalid_mask = (
            X.isin([np.inf, -np.inf]).any(axis=1) | 
            X.isnull().any(axis=1) | 
            y.isnull()
        )
        return mask_robust & mask_abs & ~invalid_mask
    
    # Helper function for comprehensive outlier filtering
    def apply_robust_outlier_filtering(df_clean, X, y, include_fallback=True):
        """
        Apply comprehensive robust outlier filtering with optional fallback.
        
        Args:
            df_clean: Clean DataFrame
            X: Feature matrix
            y: Target vector
            include_fallback: Whether to include fallback filtering
            
        Returns:
            dict: Filtering results with X_clean, y_clean, and diagnostic info
        """
        # Create robust outlier masks
        mask_robust_ocf = robust_outlier_mask(df_clean, 'earnings_surprise_ratio', lower=0.05, upper=0.95)
        mask_robust_cap = robust_outlier_mask(df_clean, 'cap_intensity', lower=0.05, upper=0.95)
        mask_abs = df_clean['earnings_surprise_ratio'].abs() < 100
        mask_robust = mask_robust_ocf & mask_robust_cap & (df_clean['rnd_to_btd'].abs() <= 100) & df_clean['rnd_to_btd'].notna()
        
        # Apply comprehensive filtering
        mask = create_robust_mask(df_clean, X, y, mask_robust, mask_abs)
        X_clean = X[mask]
        y_clean = y[mask]
        
        result = {
            'X_clean': X_clean,
            'y_clean': y_clean,
            'mask': mask,
            'rows_remaining': X_clean.shape[0],
            'fallback_used': False
        }
        
        # Apply fallback if requested and no data remains
        if include_fallback and (X_clean.empty or y_clean.empty):
            print("[WARNING] No data after robust filtering. Falling back to less strict filtering.")
            mask_fallback = (df_clean['earnings_surprise_ratio'].abs() < 100)
            X_clean_fallback = X[mask_fallback]
            y_clean_fallback = y[mask_fallback]
            
            result.update({
                'X_clean': X_clean_fallback,
                'y_clean': y_clean_fallback,
                'mask': mask_fallback,
                'rows_remaining': X_clean_fallback.shape[0],
                'fallback_used': True
            })
        
        return result
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)
        print(f"[WORKING DIRECTORY] Changed to script directory: {script_dir}")
    else:
        print(f"[WORKING DIRECTORY] Current directory: {os.getcwd()}")

    essential_caches = [
        'sp500_cache.json',
        'income_statement_cache.json',
        'balance_sheet_cache.json',
        'cash_flow_cache.json',
        EARNINGS_CALENDAR_CACHE_FILE
    ]
    cache_statuses = []
    need_api = False
    expired_caches = []
    valid_caches = []
    missing_caches = []
    print("\n[CACHE DIAGNOSTICS] Checking essential caches:")
    for cache_file in essential_caches:
        exists = os.path.exists(cache_file)
        expired = is_cache_expired(cache_file)
        cache_statuses.append((cache_file, exists, expired))
        if not exists:
            print(f"  [MISSING] {cache_file} does not exist.")
            need_api = True
            missing_caches.append(cache_file)
        elif expired:
            print(f"  [EXPIRED] {cache_file} exists but is expired.")
            need_api = True
            expired_caches.append(cache_file)
        else:
            print(f"  [VALID] {cache_file} exists and is unexpired.")
            valid_caches.append(cache_file)
    print("\n[CACHE SUMMARY]")
    print(f"  Valid caches:   {valid_caches}")
    print(f"  Expired caches: {expired_caches}")
    print(f"  Missing caches: {missing_caches}")
    API_KEY = None
    if need_api:
        print(f"\nThe following caches are missing or expired: {expired_caches + missing_caches} - API key needed to update.")
        API_KEY = get_api_key()
        if API_KEY is None:
            print("No API key provided. Cannot fetch missing data. Exiting.")
            return
        # Fetch and update ALL required caches
        print("[CACHE UPDATE] Fetching and updating all required caches...")
        # 1. S&P 500 constituents
        sp500_df = fetch_sp500_constituents(API_KEY)
        tickers = sp500_df['ticker'].tolist() if sp500_df is not None and not sp500_df.empty else []
        # 2. Financial statements
        statements_data = fetch_statements_batch(tickers, API_KEY, batch_size=50)
        # Market cap is available from company profiles, no need for separate quotes API
        # Ratios are calculated from financial statements, no need to fetch from API
        # 4. Profiles (only fetch missing from cache)
        print("[CACHE UPDATE] Fetching missing company profiles...")
        profile_cache = load_cache("profile_cache.json")
        missing_profiles = [ticker for ticker in tickers if ticker not in profile_cache]
        print(f"[CACHE UPDATE] {len(missing_profiles)} profiles missing from cache.")
        for ticker in tqdm(missing_profiles, desc="Fetching missing profiles"):
            profile = fetch_company_profile(ticker, API_KEY)
            if profile is not None:
                profile_cache[ticker] = profile
        save_cache(profile_cache, "profile_cache.json")
        print("[CACHE UPDATE] All required caches have been updated.")
    else:
        print("\n[CACHE] All required caches are present and unexpired. Proceeding with cached data only.")
        API_KEY = None  # No API key needed for cached data
        
        # Load data from cache
        sp500_df = fetch_sp500_constituents(API_KEY)  # This will load from cache
        tickers = sp500_df['ticker'].tolist() if sp500_df is not None and not sp500_df.empty else []
        
        # Load other data from cache
        statements_data = fetch_statements_batch(tickers, API_KEY, batch_size=50)
        # Fetch ratios data including effectiveTaxRate for Book-to-Tax Difference calculation
        ratios_data = fetch_ratios_in_batches(tickers, API_KEY, cache_file="ratios_cache.json")
    """
    Main function that runs the complete analysis pipeline.
    """
    print("="*60)
    print("SOFTWARE R&D CAPITALIZATION ANALYSIS")
    print("="*60)
    
    # Define features at the very top so it is available for all feature engineering steps
    # Include all control variables that will be used in the regression model
    features = ['cap_intensity', 'rnd_to_btd', 'leverage', 'revenue_growth', 'current_inventory_ratio', 
                'profitability', 'ppe_total_assets', 
                'acquisitions_intensity', 'goodwill_change_intensity', 'intangible_assets_change_intensity', 'total_dna', 'deferred_revenue_to_assets', 'log_market_cap', 'log_total_assets', 'earnings_quality', 'quarter_number']
    # Initialize time_series_metrics to avoid NameError
    time_series_metrics = {}
    # Data Collection
    
    # Default: use all S&P 500 tickers
    sp500_df = fetch_sp500_constituents(API_KEY)
    if sp500_df.empty:
        print("No S&P 500 data available. Exiting.")
        return
    tickers = sp500_df['ticker'].tolist()
    
    # Quick test mode - uncomment to test with just a few tickers
    # tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'ADBE']
    # print(f"TEST MODE: Using {len(tickers)} test tickers")
    
    print(f"Using all S&P 500 tickers: {len(tickers)} companies")
    # Ensure sp500_df is a DataFrame before using pandas methods
    if not isinstance(sp500_df, pd.DataFrame):
        sp500_df = pd.DataFrame(sp500_df)
    tech_tickers = sp500_df[sp500_df['sector'].str.lower().str.contains('tech')]['ticker'].tolist()
    
    # Load financial data from cache (no redundant API calls)
    print("Loading financial data from cache...")
    
    # Load all financial statements from cache
    statements_data = {}
    quotes_data = pd.DataFrame()
    market_cap_map = {}
    
    # Load balance sheet cache
    bs_cache = load_cache("balance_sheet_cache.json")
    is_cache = load_cache("income_statement_cache.json")
    cf_cache = load_cache("cash_flow_cache.json")
    quotes_cache = load_cache("quote_cache.json")
    
    # Process each ticker from cache
    for ticker in tickers:
        if ticker in bs_cache and ticker in is_cache and ticker in cf_cache:
            bs_data = bs_cache[ticker]
            is_data = is_cache[ticker]
            cf_data = cf_cache[ticker]
            
            if bs_data and is_data and cf_data:
                statements_data[ticker] = (bs_data, is_data, cf_data)
        
        # Load market cap from quotes cache
        if ticker in quotes_cache:
            ticker_quote = quotes_cache[ticker]
            if ticker_quote and isinstance(ticker_quote, dict):
                market_cap = ticker_quote.get('marketCap', np.nan)
                market_cap_map[ticker] = market_cap
    
    print(f"Loaded data for {len(statements_data)} tickers from cache")
    
    # Load earnings calendar data for earnings surprise ratio calculation
    print("[DATA LOADING] Loading earnings calendar data from cache...")
    earnings_calendar_data = load_cache(EARNINGS_CALENDAR_CACHE_FILE)
    if not earnings_calendar_data:
        earnings_calendar_data = {}
    print("[DATA LOADING] Earnings calendar data loaded from cache.")
    rows = []
    first_ticker_checked = False
    for ticker, item in statements_data.items():
        # Defensive: always try to coerce to DataFrame
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            print(f"[PIPELINE WARNING] {ticker}: item is not a 3-tuple/list, got type {type(item)}. Skipping.")
            continue
        bs, is_, cf = item
        # Always convert to DataFrame if not already
        if not isinstance(bs, pd.DataFrame):
            bs = pd.DataFrame(bs)
        if not isinstance(is_, pd.DataFrame):
            is_ = pd.DataFrame(is_)
        if not isinstance(cf, pd.DataFrame):
            cf = pd.DataFrame(cf)
        # Print diagnostics for the first 3 tickers
        if not first_ticker_checked or (len(rows) < 3):
            print(f"[DIAGNOSTIC] {ticker} balance sheet shape: {bs.shape}, columns: {list(bs.columns)}")
            print(f"[DIAGNOSTIC] {ticker} income statement shape: {is_.shape}, columns: {list(is_.columns)}")
            print(f"[DIAGNOSTIC] {ticker} cash flow shape: {cf.shape}, columns: {list(cf.columns)}")
            if not bs.empty:
                print(f"[DIAGNOSTIC] {ticker} balance sheet sample:\n{bs.head()}\n")
            if not is_.empty:
                print(f"[DIAGNOSTIC] {ticker} income statement sample:\n{is_.head()}\n")
            if not cf.empty:
                print(f"[DIAGNOSTIC] {ticker} cash flow sample:\n{cf.head()}\n")
            first_ticker_checked = True
        # Defensive: skip if any are None or not DataFrame after conversion
        if bs is None or is_ is None or cf is None:
            print(f"[PIPELINE WARNING] {ticker}: One or more statements is None after conversion. Skipping.")
            continue
        if not hasattr(bs, 'columns') or not hasattr(is_, 'columns') or not hasattr(cf, 'columns'):
            print(f"[PIPELINE WARNING] {ticker}: One or more statements missing 'columns' attribute after conversion. Skipping.")
            continue
        if 'date' not in bs.columns:
            print(f"[PIPELINE WARNING] {ticker}: 'date' column missing from balance sheet. Skipping.")
            continue
        ticker_rows = []
        # Defensive: ensure bs, is_, cf are DataFrames and not None
        if bs is None or is_ is None or cf is None:
            print(f"[PIPELINE WARNING] {ticker}: One or more statements is None (redundant check). Skipping.")
            continue
        if not isinstance(bs, pd.DataFrame) or 'date' not in bs.columns:
            print(f"[PIPELINE WARNING] {ticker}: Balance sheet is not DataFrame or missing 'date'. Skipping.")
            continue
        # --- FIX: Filter dataframes to the rolling window BEFORE processing each year ---
        bs = filter_to_rolling_years(bs, 'date', years=3)
        is_ = filter_to_rolling_years(is_, 'date', years=3)
        cf = filter_to_rolling_years(cf, 'date', years=3)

        if bs.empty:
            print(f"[SKIP] {ticker}: No data remaining after filtering to the rolling window.")
            continue

        # Iterate over the filtered years only
        for year in bs["date"]:
            try:
                b_row = bs[bs.date == year]
                inc_row = is_[is_["date"] == year]
                c_row = cf[cf["date"] == year]
                if b_row.empty or inc_row.empty or c_row.empty:
                    print(f"[SKIP] {ticker} {year}: b_row.empty={b_row.empty}, inc_row.empty={inc_row.empty}, c_row.empty={c_row.empty}")
                    continue
                b = b_row.iloc[0]
                inc = inc_row.iloc[0]
                c = c_row.iloc[0]
                # Helper function for safe financial calculations with pandas NaN handling
                def safe_financial_calc(data_dict, calc_dict):
                    """
                    Safely calculate financial ratios using pandas NaN handling.
                    
                    Args:
                        data_dict: Dictionary of raw financial data
                        calc_dict: Dictionary of calculation formulas
                    
                    Returns:
                        Dictionary of calculated ratios with NaN handling
                    """
                    # Convert to pandas Series for efficient NaN handling
                    data_series = pd.Series(data_dict)
                    
                    # Create a DataFrame for vectorized operations
                    df_calc = pd.DataFrame([data_series])
                    
                    # Apply calculations using pandas operations
                    results = {}
                    for name, formula in calc_dict.items():
                        try:
                            # Use pandas eval for safe evaluation
                            result = df_calc.eval(formula).iloc[0]
                            results[name] = result
                        except Exception:
                            results[name] = np.nan
                    
                    # Convert back to regular dictionary and handle infinities
                    results_series = pd.Series(results)
                    results_series = results_series.replace([np.inf, -np.inf], np.nan)
                    
                    return results_series.to_dict()
                
                # Real Estate filter removed - all companies now use actual R&D expense values
                
                # --- ROLLING 4-QUARTER CALCULATIONS ---
                # Prepare data for rolling calculations
                bs_full = bs.sort_values('date').reset_index(drop=True)
                is_full = is_.sort_values('date').reset_index(drop=True)
                cf_full = cf.sort_values('date').reset_index(drop=True)
                
                # Find current index for rolling calculations
                current_idx = bs_full[bs_full['date'] == year].index[0] if not bs_full[bs_full['date'] == year].empty else None
                
                if current_idx is not None and current_idx >= 3:
                    # Direct calculate-and-assign for rolling 4Q variables
                    # Income statement variables (4Q sum)
                    rnd_exp_4q = rolling_4q_sum(is_full.get('researchAndDevelopmentExpenses', pd.Series(dtype=float)), current_idx)
                    ni_4q = rolling_4q_sum(is_full.get('netIncome', pd.Series(dtype=float)), current_idx)
                    rev_4q = rolling_4q_sum(is_full.get('revenue', pd.Series(dtype=float)), current_idx)
                    ebitda_4q = rolling_4q_sum(is_full.get('ebitda', pd.Series(dtype=float)), current_idx)
                    gross_profit_4q = rolling_4q_sum(is_full.get('grossProfit', pd.Series(dtype=float)), current_idx)
                    dna_4q = rolling_4q_sum(is_full.get('depreciationAndAmortization', pd.Series(dtype=float)), current_idx)
                    income_before_tax_4q = rolling_4q_sum(is_full.get('incomeBeforeTax', pd.Series(dtype=float)), current_idx)
                    income_tax_expense_4q = rolling_4q_sum(is_full.get('incomeTaxExpense', pd.Series(dtype=float)), current_idx)
                    
                    # Cash flow variables (4Q sum)
                    ocf_4q = rolling_4q_sum(cf_full.get('operatingCashFlow', pd.Series(dtype=float)), current_idx)
                    capex_4q = rolling_4q_sum(cf_full.get('capitalExpenditure', pd.Series(dtype=float)), current_idx)
                    acquisitions_4q = rolling_4q_sum(cf_full.get('acquisitionsNet', pd.Series(dtype=float)), current_idx)
                    opex_4q = rolling_4q_sum(cf_full.get('operatingExpenses', pd.Series(dtype=float)), current_idx)
                    
                    # Balance sheet variables (4Q average)
                    assets_4q = rolling_4q_avg(bs_full.get('totalAssets', pd.Series(dtype=float)), current_idx)
                    debt_4q = rolling_4q_avg(bs_full.get('totalLiabilities', pd.Series(dtype=float)), current_idx)
                    goodwill_4q = rolling_4q_avg(bs_full.get('goodwill', pd.Series(dtype=float)), current_idx)
                    inventory_4q = rolling_4q_avg(bs_full.get('inventory', pd.Series(dtype=float)), current_idx)
                    total_current_assets_4q = rolling_4q_avg(bs_full.get('totalCurrentAssets', pd.Series(dtype=float)), current_idx)
                    intangible_assets_4q = rolling_4q_avg(bs_full.get('intangibleAssets', pd.Series(dtype=float)), current_idx)
                    deferred_revenue_4q = rolling_4q_avg(bs_full.get('deferredRevenue', pd.Series(dtype=float)), current_idx)
                    
                    # Change variables (4Q delta)
                    intangible_change_4q = rolling_4q_delta(bs_full.get('intangibleAssets', pd.Series(dtype=float)), current_idx)
                    goodwill_change_4q = rolling_4q_delta(bs_full.get('goodwill', pd.Series(dtype=float)), current_idx)
                    ppe_change_4q = rolling_4q_delta(bs_full.get('propertyPlantEquipmentNet', pd.Series(dtype=float)), current_idx)
                else:
                    # Not enough data for rolling calculations
                    rnd_exp_4q = ni_4q = rev_4q = ocf_4q = capex_4q = acquisitions_4q = opex_4q = ebitda_4q = gross_profit_4q = dna_4q = np.nan
                    assets_4q = debt_4q = goodwill_4q = inventory_4q = total_current_assets_4q = intangible_assets_4q = np.nan
                    intangible_change_4q = goodwill_change_4q = ppe_change_4q = np.nan
                    deferred_revenue_4q = np.nan
                    income_before_tax_4q = income_tax_expense_4q = np.nan
                # ... existing code ...
                # Extract all financial variables at once (using rolling 4-quarter values)
                financial_data = {
                    'capex': capex_4q,  # Rolling 4-quarter capital expenditure
                    'rnd_exp': rnd_exp_4q,  # Rolling 4-quarter R&D expense
                    'ocf': ocf_4q,  # Rolling 4-quarter operating cash flow
                    'ni': ni_4q,  # Rolling 4-quarter net income
                    'assets': assets_4q,  # Rolling 4-quarter average total assets
                    'debt': debt_4q,  # Rolling 4-quarter average total liabilities
                    'rev': rev_4q,  # Rolling 4-quarter revenue
                    'ppe': ppe_change_4q,  # Rolling 4-quarter delta PPE
                    'ebitda': ebitda_4q,  # Rolling 4-quarter EBITDA
                    'acquisitions_net': acquisitions_4q,  # Rolling 4-quarter net acquisitions
                    'goodwill': goodwill_4q,  # Rolling 4-quarter average goodwill
                    'gross_profit': gross_profit_4q,  # Rolling 4-quarter gross profit
                    'inventory': inventory_4q,  # Rolling 4-quarter average inventory
                    'total_current_assets': total_current_assets_4q,  # Rolling 4-quarter average current assets
                    'intangible_assets': intangible_assets_4q,  # Rolling 4-quarter average intangible assets
                    'goodwill_change_4q': goodwill_change_4q,  # Rolling 4-quarter delta
                    'intangible_change_4q': intangible_change_4q,  # Rolling 4-quarter delta
                    'opex_4q': opex_4q,  # Rolling 4-quarter operating expenses
                    'dna': dna_4q,  # Rolling 4-quarter depreciation and amortization
                    'income_before_tax': income_before_tax_4q,  # Rolling 4-quarter income before tax
                    'income_tax_expense': income_tax_expense_4q,  # Rolling 4-quarter income tax expense
                    'effective_tax_rate': 0.21,  # Default US corporate tax rate
                    'deferred_revenue': deferred_revenue_4q,  # Rolling 4-quarter average deferred revenue
                    'market_cap': market_cap_map.get(ticker, np.nan)  # Market capitalization
                }
                
                # Real Estate filter removed - no diagnostic print needed
                
                # Extract quarter number from period field for quarter control variable
                quarter_number = None
                if 'period' in inc:
                    period_str = str(inc['period'])
                    if period_str.startswith('Q'):
                        try:
                            quarter_number = int(period_str[1:])  # Extract number after 'Q'
                        except ValueError:
                            quarter_number = None
                
                # Extract year from date for year control variable
                year_number = None
                try:
                    year_date = pd.to_datetime(year)
                    year_number = year_date.year
                except (ValueError, TypeError):
                    year_number = None
                
                # Check critical variables globally using pandas
                critical_vars = ['capex', 'rnd_exp', 'ni', 'assets', 'intangible_assets']
                critical_series = pd.Series({k: financial_data[k] for k in critical_vars})
                # Only skip if rnd_exp is NaN (not if it is zero), but still skip if ni is NaN or zero
                # Only skip if ALL critical variables are NaN (indicating no financial statements)
                if critical_series.isna().all():
                    # Remove per-row [SKIP] print, just count skips
                    if 'skip_counts' not in locals():
                        skip_counts = {}
                    skip_counts[ticker] = skip_counts.get(ticker, 0) + 1
                    continue
                
                # Calculate goodwill change from previous period
                try:
                    bs_dates = pd.to_datetime(bs['date'])
                    current_date = pd.to_datetime(year)
                    prev_mask = bs_dates < current_date
                    prev_bs = bs.loc[prev_mask]
                    prev_b_row = prev_bs.sort_values(by="date").iloc[-1] if not prev_bs.empty else None
                    financial_data['prev_goodwill'] = prev_b_row.get("goodwill", np.nan) if prev_b_row is not None else np.nan
                except Exception:
                    financial_data['prev_goodwill'] = np.nan
                
                # Calculate intangible assets change from previous period
                try:
                    prev_b_row = prev_bs.sort_values(by="date").iloc[-1] if not prev_bs.empty else None
                    financial_data['prev_intangible_assets'] = prev_b_row.get("intangibleAssets", np.nan) if prev_b_row is not None else np.nan
                except Exception:
                    financial_data['prev_intangible_assets'] = np.nan
                
                # Calculate derived metrics using vectorized operations
                financial_data['goodwill_change'] = financial_data['goodwill_change_4q']
                financial_data['intangible_assets_change'] = financial_data['intangible_change_4q']
                
                # Define all financial ratios in one place for clarity and consistency
                ratio_calculations = {
                    # Core treatment variables
                    'cap_intensity': 'capex / rnd_exp',  # Capitalization intensity (Capex / R&D)
            
                    'rnd_to_btd': 'rnd_exp / (income_before_tax - (income_tax_expense / effective_tax_rate))',  # R&D expense / Book-to-Tax Difference
                    
                    # Control variables
                    'ocf_margin': 'ocf / rev',  # Operating cash flow margin (OCF / Revenue)
                    'earnings_quality': 'ocf / ni',  # Earnings quality (OCF / Net Income)
                    'leverage': 'debt / assets',  # Leverage ratio
                    'current_inventory_ratio': 'inventory / total_current_assets',  # Current inventory ratio
                    'ppe_total_assets': 'ppe / assets',  # PPE to total assets
                    # 'rnd_intensity': 'rnd_exp / rev',  # R&D intensity (R&D / Revenue) - REMOVED
                    'acquisitions_intensity': 'abs(acquisitions_net) / assets',  # Acquisitions intensity
                    'goodwill_change_intensity': 'abs(goodwill_change_4q) / intangible_assets',  # Goodwill change intensity (4Q delta goodwill / 4Q average intangible assets)
                    'intangible_assets_change_intensity': 'intangible_assets_change / assets',  # Intangible assets change intensity
                    'total_dna': 'dna / assets',  # Total D&A to assets ratio (4Q sum D&A / 4Q average assets)
                    'deferred_revenue_to_assets': 'deferred_revenue / assets',  # Deferred revenue to total assets ratio
                    'profitability': 'gross_profit / assets'  # Profitability
                }
                
                # CRITICAL FIX: Data quality validation before ratio calculations
                # Reject companies with obviously corrupted financial data
                def validate_financial_data_quality(financial_data, ticker, year):
                    """
                    Validate financial data quality and reject corrupted data.
                    This prevents extreme outliers from corrupting the analysis.
                    """
                    warnings = []
                    
                    # Define reasonable bounds for financial metrics
                    bounds = {
                        'revenue': (1e6, 1e12),      # $1M to $1T
                        'assets': (1e6, 1e12),       # $1M to $1T  
                        'capex': (-1e11, 1e11),      # -$100B to $100B
                        'rnd_exp': (0, 1e11),        # $0 to $100B
                        'ocf': (-1e11, 1e11),        # -$100B to $100B
                        'ni': (-1e11, 1e11),         # -$100B to $100B
                    }
                    
                    for metric, (min_val, max_val) in bounds.items():
                        if metric in financial_data:
                            value = financial_data[metric]
                            if not pd.isna(value):
                                if value < min_val or value > max_val:
                                    warnings.append(f"{metric}: {value:.0f} outside bounds [{min_val:.0e}, {max_val:.0e}]")
                    
                    # If any critical metrics are corrupted, reject this data point
                    critical_metrics = ['revenue', 'assets', 'capex', 'rnd_exp']
                    corrupted_critical = [w for w in warnings if any(metric in w for metric in critical_metrics)]
                    
                    if corrupted_critical:
                        print(f"[DATA QUALITY] {ticker} {year}: REJECTED due to corrupted data:")
                        for warning in corrupted_critical:
                            print(f"  {warning}")
                        return False, warnings
                    
                    return True, warnings
                
                # Validate data quality before proceeding
                data_is_valid, quality_warnings = validate_financial_data_quality(financial_data, ticker, year)
                
                if not data_is_valid:
                    # Skip this data point - it's corrupted
                    continue
                
                # Use helper function for safe calculations with pandas NaN handling
                ratios = safe_financial_calc(financial_data, ratio_calculations)
                
                # Create data row with all calculated ratios
                ticker_rows.append({
                    "ticker": ticker,
                    "date": year,
                    "quarter_number": quarter_number,  # Quarter control variable (1-4)
                    "year": year_number,  # Year control variable (4-digit integer)
                    
                    # Raw financial data
                    "revenue": financial_data['rev'],
                    "assets": financial_data['assets'],
                    "rnd_expense": financial_data['rnd_exp'],
                    "capex": financial_data['capex'],
                    "intangible_assets_change": financial_data['intangible_assets_change'],
                    "operating_cash_flow": financial_data['ocf'],
                    "net_income": financial_data['ni'],
                    
                    # --- Add missing critical columns ---
                    "total_debt": financial_data.get('total_debt', np.nan),
                    "property_plant_equipment_net": financial_data.get('property_plant_equipment_net', np.nan),
                    "intangible_assets": financial_data.get('intangible_assets', np.nan),
                    "goodwill_change_4q": financial_data.get('goodwill_change_4q', np.nan),
                    "intangible_change_4q": financial_data.get('intangible_change_4q', np.nan),
                    "income_before_tax": financial_data.get('income_before_tax', np.nan),
                    "income_tax_expense": financial_data.get('income_tax_expense', np.nan),
                    "effective_tax_rate": financial_data.get('effective_tax_rate', np.nan),
                    "market_cap": financial_data['market_cap'],
                    
                    # Calculated ratios (all from the ratio_calculations dictionary)
                    "cap_intensity": ratios['cap_intensity'],
                    
                    "rnd_to_btd": ratios['rnd_to_btd'],
                    "ocf_margin": ratios['ocf_margin'],
                    "earnings_quality": ratios['earnings_quality'],
                    "leverage": ratios['leverage'],
                    "current_inventory_ratio": ratios['current_inventory_ratio'],
                    "ppe_total_assets": ratios['ppe_total_assets'],
                    # "rnd_intensity": ratios['rnd_intensity'],  # REMOVED
                    "acquisitions_intensity": ratios['acquisitions_intensity'],
                    "goodwill_change_intensity": ratios['goodwill_change_intensity'],
                    "intangible_assets_change_intensity": ratios['intangible_assets_change_intensity'],
                    "total_dna": ratios['total_dna'],
                    "deferred_revenue_to_assets": ratios['deferred_revenue_to_assets'],
                    "profitability": ratios['profitability'],
                    
                    # Earnings surprise ratio (will be calculated from earnings calendar data)
                    "earnings_surprise_ratio": np.nan
                })
            except Exception as e:
                print(f"[ERROR] {ticker} {year}: {e}")
                continue
        rows.extend(ticker_rows)

    # After collecting rows, if len(rows) == 0, print a summary diagnostic before raising ValueError
    if len(rows) == 0:
        print("[SUMMARY DIAGNOSTIC] No financial statement data was collected for any ticker.")
        print("Possible reasons: API key invalid, expired, restricted, endpoint down, or network issue.")
        print("Check the above diagnostics for details.")
        # --- BEGIN DIAGNOSTICS ---
        cache_files = [
            'income_statement_cache.json',
            'balance_sheet_cache.json',
            'cash_flow_cache.json',
            'quote_cache.json',
            'historical_price_cache.json',
            'sp500_cache.json'
        ]
        # Print first 10 tickers being processed
        if 'tickers' in locals():
            print(f"Tickers being processed (first 10): {tickers[:10]}")
        for fname in cache_files:
            exists = os.path.exists(fname)
            expired = is_cache_expired(fname) if exists else 'N/A'
            size = os.path.getsize(fname) if exists else 0
            print(f"  - {fname}: exists={exists}, expired={expired}, size={size}")
            if exists and not expired:
                try:
                    cache = load_cache(fname)
                    if isinstance(cache, dict):
                        keys = list(cache.keys())
                        print(f"    First 10 keys: {keys[:10]}")
                        # Print missing tickers if any
                        if 'tickers' in locals():
                            missing = [t for t in tickers if t not in keys]
                            if missing:
                                print(f"    [ERROR] Missing tickers in {fname}: {missing[:10]} (showing up to 10)")
                    elif isinstance(cache, list):
                        print(f"    Cache is a list with {len(cache)} items.")
                    else:
                        print(f"    Cache is type {type(cache)}")
                except Exception as e:
                    print(f"    [CACHE READ ERROR] {e}")
        print(f"  - API key set: {bool(api_key) if 'api_key' in locals() else False}")
        print(f"  - Cache-only mode: {cache_only_mode if 'cache_only_mode' in locals() else 'N/A'}")
        # Prompt for API key if not set
        if 'api_key' in locals() and not api_key:
            api_key = input("Enter your FMP API key: ")
        print("  - If caches are present and not expired, but still no data, check cache contents for missing tickers or corrupted data.")
        # --- END DIAGNOSTICS ---
        raise ValueError("No financial data was retrieved for any ticker. Please check your API key, ticker list, or network connection.")

    # After collecting rows, print aggregate skip summary before regression analysis
    if 'skip_counts' in locals():
        total_skipped = sum(skip_counts.values())
        print(f"[SUMMARY] Skipped {total_skipped} rows due to missing critical variables (R&D, net income, etc.)")
        print(f"[SUMMARY] Skipped rows per ticker (top 10): {dict(sorted(skip_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")



    # Data Processing and Cleaning
    # ============================
    print(f"\nCollected {len(rows)} data points")

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(rows)
    
    # Calculate earnings surprise ratio from earnings calendar data
    print("[DATA PROCESSING] Calculating earnings surprise ratios...")
    for ticker in df['ticker'].unique():
        if ticker in earnings_calendar_data and earnings_calendar_data[ticker]:
            ticker_mask = df['ticker'] == ticker
            for idx in df[ticker_mask].index:
                date = df.loc[idx, 'date']
                earnings_surprise = calculate_earnings_surprise_ratio(earnings_calendar_data[ticker], date)
                df.loc[idx, 'earnings_surprise_ratio'] = earnings_surprise
    print(f"[DATA PROCESSING] Earnings surprise ratios calculated for {df['earnings_surprise_ratio'].notna().sum()} data points")

        # --- DIAGNOSTIC: rnd_to_btd calculation ---
    if 'rnd_to_btd' in df.columns:
        print("\n[DIAGNOSTIC] rnd_to_btd calculation sample:")
        sample = df[['ticker','date','rnd_expense','income_before_tax','income_tax_expense','effective_tax_rate','rnd_to_btd']].head(10)
        print(sample)
        btd = df['income_before_tax'] - (df['income_tax_expense'] / df['effective_tax_rate'])
        print(f"rnd_to_btd denominator (book-to-tax difference) stats: min={btd.min()}, max={btd.max()}, zero count={(btd==0).sum()}, nan count={btd.isna().sum()}")
        print(f"rnd_to_btd NaN count: {df['rnd_to_btd'].isna().sum()}")
        print(f"rnd_to_btd <0 count: {(df['rnd_to_btd']<0).sum()}")
        print(f"rnd_to_btd >100 count: {(df['rnd_to_btd']>100).sum()}")
        
        # --- FIX: Recalculate rnd_to_btd for NaN values ---
        mask_nan = df['rnd_to_btd'].isna()
        num_nan = mask_nan.sum()
        if num_nan > 0:
            fixed = df.loc[mask_nan, 'rnd_expense'] / (df.loc[mask_nan, 'income_before_tax'] - (df.loc[mask_nan, 'income_tax_expense'] / df.loc[mask_nan, 'effective_tax_rate']))
            df.loc[mask_nan, 'rnd_to_btd'] = fixed
            print(f"[FIX] Recalculated rnd_to_btd for {num_nan} rows with NaN. Now NaN count: {df['rnd_to_btd'].isna().sum()}")

    initial_shape = df.shape
    # --- FIX: Apply the rolling window filter HERE, after all calculations are done ---
    print("\n=== APPLYING ROLLING 3-YEAR WINDOW FILTER (POST-CALCULATION) ===")
    df = filter_to_rolling_years(df, 'date', years=3)
    print(f"Data after final rolling window filter: {len(df)} observations")

    # Now, proceed with cleaning on the correctly filtered DataFrame
    # --- FIX: Corrected column names to match the DataFrame ---
    critical_cols = [
        'capex', 'rnd_expense', 'operating_cash_flow', 'net_income', 'assets',
        'total_debt', 'revenue', 'property_plant_equipment_net', 'intangible_assets',
        'goodwill_change_4q', 'intangible_change_4q'
    ]
    print(f"[DIAGNOSTIC] DataFrame shape before dropna: {df.shape}")
    print(f"[DIAGNOSTIC] Columns present: {list(df.columns)}")

    # Check which critical columns are actually in the DataFrame before trying to use them
    existing_critical_cols = [col for col in critical_cols if col in df.columns]
    missing_critical_cols = [col for col in critical_cols if col not in df.columns]
    if missing_critical_cols:
        print(f"[WARNING] The following critical columns are missing from the DataFrame and will be skipped: {missing_critical_cols}")
    if existing_critical_cols:
        missing_counts = df[existing_critical_cols].isna().sum()
        print(f"[DIAGNOSTIC] Missing values in critical columns before dropna:\n{missing_counts}")
        # Fill NaN with zero for all critical columns
        df[existing_critical_cols] = df[existing_critical_cols].fillna(0)
        df = df.fillna(0)
        print(f"[DIAGNOSTIC] DataFrame shape after dropna: {df.shape}")
    else:
        print("[ERROR] No critical columns found in DataFrame!")
    if df.empty:
        print("[ERROR] All rows dropped after dropna on critical columns!")
        print(f"[DIAGNOSTIC] Columns in DataFrame: {list(df.columns)}")
        raise ValueError("No financial data was retrieved for any ticker. Please check your API key, ticker list, or network connection.")

    # === VALUE RANGE DIAGNOSTICS ===
    print("\n=== VALUE RANGE DIAGNOSTICS ===")
    print("Checking if financial ratios are in reasonable ranges...")
    
    # Check cap_intensity (intangible assets / R&D expenses)
    if 'cap_intensity' in df.columns:
        cap_intensity_stats = df['cap_intensity'].describe()
        print(f"\ncap_intensity statistics:")
        print(f"  Count: {cap_intensity_stats['count']}")
        print(f"  Mean: {cap_intensity_stats['mean']:.2f}")
        print(f"  Std: {cap_intensity_stats['std']:.2f}")
        print(f"  Min: {cap_intensity_stats['min']:.2f}")
        print(f"  Max: {cap_intensity_stats['max']:.2f}")
        print(f"  25%: {cap_intensity_stats['25%']:.2f}")
        print(f"  50%: {cap_intensity_stats['50%']:.2f}")
        print(f"  75%: {cap_intensity_stats['75%']:.2f}")
        
        # Check for extreme values
        extreme_cap = df[df['cap_intensity'] > 100]
        if len(extreme_cap) > 0:
            print(f"  WARNING: {len(extreme_cap)} observations have cap_intensity > 100")
            print(f"  These may be causing filtering issues. Sample extreme values:")
            if not isinstance(extreme_cap, pd.DataFrame):
                extreme_cap = pd.DataFrame(extreme_cap)
            print(extreme_cap[['ticker', 'date', 'cap_intensity']].head())
    
    # Check ocf_margin (operating cash flow / revenue)
    if 'ocf_margin' in df.columns:
        ocf_margin_stats = df['ocf_margin'].describe()
        print(f"\nocf_margin statistics (OCF / Revenue):")
        print(f"  Count: {ocf_margin_stats['count']}")
        print(f"  Mean: {ocf_margin_stats['mean']:.2f}")
        print(f"  Std: {ocf_margin_stats['std']:.2f}")
        print(f"  Min: {ocf_margin_stats['min']:.2f}")
        print(f"  Max: {ocf_margin_stats['max']:.2f}")
        print(f"  25%: {ocf_margin_stats['25%']:.2f}")
        print(f"  50%: {ocf_margin_stats['50%']:.2f}")
        print(f"  75%: {ocf_margin_stats['75%']:.2f}")
        
        # Check for extreme values
        extreme_ocf = df[df['ocf_margin'] > 10]
        if len(extreme_ocf) > 0:
            print(f"  WARNING: {len(extreme_ocf)} observations have ocf_margin > 10")
            print(f"  These may be causing filtering issues. Sample extreme values:")
            if not isinstance(extreme_ocf, pd.DataFrame):
                extreme_ocf = pd.DataFrame(extreme_ocf)
            print(extreme_ocf[['ticker', 'date', 'ocf_margin']].head())
    

    
    # Check rnd_to_btd (R&D expenses / Book-to-Tax Difference)
    if 'rnd_to_btd' in df.columns:
        rnd_to_btd_stats = df['rnd_to_btd'].describe()
        print(f"\nrnd_to_btd statistics:")
        print(f"  Count: {rnd_to_btd_stats['count']}")
        print(f"  Mean: {rnd_to_btd_stats['mean']:.4f}")
        print(f"  Std: {rnd_to_btd_stats['std']:.4f}")
        print(f"  Min: {rnd_to_btd_stats['min']:.4f}")
        print(f"  Max: {rnd_to_btd_stats['max']:.4f}")
        print(f"  25%: {rnd_to_btd_stats['25%']:.4f}")
        print(f"  50%: {rnd_to_btd_stats['50%']:.4f}")
        print(f"  75%: {rnd_to_btd_stats['75%']:.4f}")
        
        # Check for extreme values
        invalid_rnd_btd = df[(df['rnd_to_btd'].abs() > 100) | df['rnd_to_btd'].isna()]
        if len(invalid_rnd_btd) > 0:
            print(f"  WARNING: {len(invalid_rnd_btd)} observations have rnd_to_btd with extreme values (>100) or NaN")
            print(f"  These may be causing filtering issues. Sample invalid values:")
            if not isinstance(invalid_rnd_btd, pd.DataFrame):
                invalid_rnd_btd = pd.DataFrame(invalid_rnd_btd)
            print(invalid_rnd_btd[['ticker', 'date', 'rnd_to_btd']].head())
    
    # Add sector information from S&P 500 data
    print("Adding sector information for industry fixed effects...")
    # Ensure both DataFrames are pandas DataFrames before merging
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if not isinstance(sp500_df, pd.DataFrame):
        sp500_df = pd.DataFrame(sp500_df)
    df = df.merge(sp500_df[['ticker', 'sector', 'sub_sector']], on='ticker', how='left')

    # Handle missing sector information
    missing_sector = df['sector'].isna().sum()
    empty_sector = (df['sector'] == '').sum()
    total_missing = missing_sector + empty_sector
    if total_missing > 0:
        print(f"Warning: {total_missing} observations have missing or empty sector information")
        print(f"  - NaN sectors: {missing_sector}")
        print(f"  - Empty string sectors: {empty_sector}")
        # Remove companies with missing or empty sector information
        df = df[df['sector'].notna() & (df['sector'] != '')]
        print(f"Removed {total_missing} observations with missing sector information")
        print(f"Remaining observations: {len(df)}")

    # --- NEW LOGIC: Create sector_baseline for industry fixed effects ---
    # Use the first industry alphabetically as the baseline
    def assign_sector_baseline(row):
        return row['sector']
    df['sector_baseline'] = df.apply(assign_sector_baseline, axis=1)
    
    # Get the first industry alphabetically as baseline
    unique_sectors = sorted(df['sector_baseline'].unique())
    baseline_sector = unique_sectors[0] if unique_sectors else 'Technology'
    print(f"Using '{baseline_sector}' as baseline sector (first alphabetically)")
    print("\nsector_baseline distribution:")
    print(df['sector_baseline'].value_counts())

    # Print sector distribution
    print("Sector distribution in the dataset:")
    sector_counts = df['sector'].value_counts()
    print(sector_counts)
    
    # Print detailed subsector counts for analysis
    print("\n=== SUBSECTOR COUNTS ===")
    subsector_counts = df['sector_baseline'].value_counts()
    for subsector, count in subsector_counts.items():
        print(f"{subsector}: {count} companies")
    print(f"Total companies in analysis: {len(df)}")
    print("=" * 50)
    
    # Verify GICS sector labels are being used
    print("\nVerifying GICS sector labels from S&P 500 API:")
    if not sp500_df.empty and 'sector' in sp500_df.columns:
        unique_sectors = sp500_df['sector'].unique()
        print(f"Unique sectors from S&P 500 API: {unique_sectors}")
        print("Expected GICS sectors: Technology, Healthcare, Financial Services, Consumer Cyclical, etc.")
    else:
        print("Warning: No sector information available from S&P 500 API")
    
    # Verify quarter control variable
    print("\nVerifying quarter control variable:")
    if 'quarter_number' in df.columns:
        quarter_counts = df['quarter_number'].value_counts().sort_index()
        print(f"Quarter distribution: {quarter_counts.to_dict()}")
        missing_quarters = df['quarter_number'].isna().sum()
        print(f"Missing quarter numbers: {missing_quarters}")
    else:
        print("Warning: quarter_number column not found in dataset")
    
    # Sort the data by ticker and date for easier analysis
    if 'ticker' in df.columns and 'date' in df.columns:
        df = df.sort_values(['ticker','date'])
    print(f"Dataset shape after initial cleaning: {df.shape}")
    print(df.isnull().sum())

    # Feature Engineering
    print('Feature engineering: starting with shape', df.shape)
    
    # Calculate year-over-year growth rates
    df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change()
    
    # Calculate earnings surprise ratio from earnings calendar data
    # This will be populated from earnings calendar data later
    df['earnings_surprise_ratio'] = np.nan
    
    # Calculate OCF TTM
    df['ocf_ttm'] = df.groupby('ticker')['operating_cash_flow'].rolling(window=4, min_periods=1).sum().reset_index(level=0, drop=True)
    
    # Note: def_rev_ratio calculation removed - deferred_revenue_to_assets now properly implemented using existing cache data
    
    # Calculate revenue volatility
    df['revenue_volatility'] = df.groupby('ticker')['revenue'].rolling(window=4, min_periods=1).std().reset_index(level=0, drop=True)
    
    # Add log transformations for size control variables
    df['log_market_cap'] = np.log(df['market_cap'].replace(0, np.nan)).fillna(0)
    df['log_total_assets'] = np.log(df['assets'].replace(0, np.nan)).fillna(0)
    
    # Feature Engineering: Add cap_costs_pct_opex if sga_expense is available
    if 'sga_expense' in df.columns:
        df['cap_costs_pct_opex'] = df['intangible_assets'] / (df['rnd_expense'] + df['sga_expense'])
        # Add to features list if it was calculated
        if 'cap_costs_pct_opex' not in features:
            features.append('cap_costs_pct_opex')

    # Outlier Detection and Removal
    print("\n=== OUTLIER DETECTION ===")
    df_clean = df.copy()  # Initialize df_clean
    df_companies = df_clean.copy()
    try:
        if not df.empty and all(col in df.columns for col in features):
            # Use pandas DataFrame for z-score calculation
            z_scores_data = scipy.stats.zscore(df[features].fillna(0), nan_policy='omit')
            # Ensure z_scores_data is 2D and columns is pd.Index
            if isinstance(z_scores_data, np.ndarray) and z_scores_data.ndim == 2:
                z_scores = pd.DataFrame(z_scores_data, columns=pd.Index(features), index=df.index)
            abs_z_scores = np.abs(z_scores)
            outlier_rows = (abs_z_scores > 5).any(axis=1)
            print(f"Found {outlier_rows.sum()} outliers out of {len(df)} observations (z > 5)")
            df_clean = df.loc[~outlier_rows].copy()
            print(f"Dataset shape after outlier removal: {df_clean.shape}")
    except Exception as e:
        print(f"Error during z-score calculation: {e}. Using original dataset.")
        df_clean = df.copy()

    # Apply rolling 3-year window filter to ensure consistent date range
    # NOTE: The time range logic accounts for lookback variables:
    # - Excludes the most recent period so all variables can regress to a next period OCF that exists
    # - Ensures the first period has a previous period for "change from previous period" variables
    # - API fetching gets 22 periods (instead of 20) to provide the extra lookback periods needed
    # TEMPORARY: Using 3-year window until Q2 2025 data becomes available
    print("\n=== APPLYING ROLLING 3-YEAR WINDOW FILTER (ADJUSTED FOR LOOKBACK VARIABLES) ===")
    df_clean = filter_to_rolling_years(df_clean, 'date', years=3)
    print(f"Dataset shape after 3-year window filter: {df_clean.shape}")

    # Only drop rows for missing values in the target and essential features
    essential_cols = ['cap_intensity', 'rnd_to_btd', 'earnings_surprise_ratio']
    print(f"Missing values in essential columns: {df_clean[essential_cols].isnull().sum().sum()} total")
    df_clean = df_clean.fillna(0)
    print(f"Shape after dropping rows with missing essential columns: {df_clean.shape}")

    # Control group logic: Tech sector dominance
    sp500_df_raw = fetch_sp500_constituents(API_KEY)
    if sp500_df_raw is None:
        sp500_df = pd.DataFrame()
    else:
        sp500_df = ensure_pd_type(sp500_df_raw, 'dataframe')
    df_clean = ensure_pd_type(df_clean, 'dataframe')
    tech_tickers = []
    if isinstance(sp500_df, pd.DataFrame) and not sp500_df.empty and 'sector' in sp500_df.columns:
        sp500_df['sector'] = sp500_df['sector'].astype(str)
        tech_mask = sp500_df['sector'].str.lower().str.contains('tech')
        tech_tickers = sp500_df[tech_mask]['ticker'].tolist()
    eligible_controls = df_clean[(df_clean['cap_intensity'] > 0) | (df_clean['rnd_to_btd'].abs() <= 100)] if isinstance(df_clean, pd.DataFrame) and not df_clean.empty else pd.DataFrame()
    eligible_controls = ensure_pd_type(eligible_controls, 'dataframe')
    if isinstance(eligible_controls, pd.DataFrame) and 'ticker' in eligible_controls.columns and tech_tickers:
        tech_controls = eligible_controls[eligible_controls['ticker'].isin(tech_tickers)]
        nontech_controls = eligible_controls[~eligible_controls['ticker'].isin(tech_tickers)]


    # Prepare data for regression
    X = df_clean[features]
    y = df_clean['earnings_surprise_ratio']
    # Ensure X and y are pandas DataFrames/Series for .head()
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=pd.Index(features))
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        y = pd.Series(y)
    print("Sample of regression DataFrame:")
    print(X.head())
    print(y.head())

    # Add constant term for regression (intercept)
    X_with_const = sm.add_constant(X)

    # Remove any rows with NaN or inf in X_with_const or y
    mask = np.isfinite(X_with_const).all(axis=1) & np.isfinite(y)
    X_with_const = X_with_const[mask]
    y = y[mask]

    if X_with_const.shape[0] == 0 or y.shape[0] == 0:
        print("No data available for regression after removing NaN/inf values.")
        return

    # Ensure y and X_with_const have the same index for statsmodels
    if isinstance(X_with_const, np.ndarray):
        X_with_const = pd.DataFrame(X_with_const, columns=pd.Index(['const'] + features))
    if isinstance(X_with_const, pd.DataFrame):
        X_with_const = X_with_const.reset_index(drop=True)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)

    # Debug: Check if regression data is empty
    if not isinstance(X, (pd.DataFrame, pd.Series)) or not isinstance(y, (pd.Series, pd.DataFrame)) or X.shape[0] == 0 or y.shape[0] == 0:
        print("No data available for regression after cleaning. Check data quality and filtering steps.")
        print("Rows remaining:", len(X) if hasattr(X, '__len__') else 0)
        print("Missing values by column:")
        # Ensure df_clean is a DataFrame for .isnull()
        if not isinstance(df_clean, pd.DataFrame):
            df_clean = pd.DataFrame(df_clean)
        if isinstance(df_clean, pd.DataFrame):
            try:
                print(df_clean[features + ['earnings_surprise_ratio']].isnull().sum())
            except Exception:
                print("[DIAGNOSTIC] Could not compute missing values summary.")
        print("Consider relaxing filtering or imputing missing values for less critical features.")
        return  # or skip regression

    # === DIAGNOSTICS: Check for degenerate regressors/target ===
    print('\n=== REGRESSION INPUT DIAGNOSTICS ===')
    # Ensure X and y are pandas DataFrames/Series for diagnostics
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=pd.Index(features))
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    # Reset indices to ensure alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    for col in features:
        var = X[col].var()
        nunique = X[col].nunique()
        print(f"Regressor '{col}': variance={var:.6g}, unique values={nunique}")
        if var == 0 or nunique == 1:
            print(f"  WARNING: '{col}' is constant in regression sample!")
        # Check if regressor is identical to target using numpy arrays to avoid index issues
        if np.array_equal(np.array(X[col].values), np.array(y.values)):
            print(f"  WARNING: '{col}' is IDENTICAL to the target variable!")
    # Check for identical regressors
    for i, col1 in enumerate(features):
        for j, col2 in enumerate(features):
            if i < j and np.array_equal(np.array(X[col1].values), np.array(X[col2].values)):
                print(f"  WARNING: '{col1}' and '{col2}' are IDENTICAL!")
    # Target diagnostics
    y_var = y.var()
    y_nunique = y.nunique()
    print(f"Target '{y.name}': variance={y_var:.6g}, unique values={y_nunique}")
    if y_var == 0 or y_nunique == 1:
        print(f"  WARNING: Target '{y.name}' is constant!")

    # Apply improved outlier detection before fitting the model
    print("\n=== IMPROVED OUTLIER DETECTION ===")
    outlier_mask, outlier_info = improved_outlier_detection(df_clean, features, 'earnings_surprise_ratio', method='isolation_forest')
    
    # Remove outliers for model fitting
    df_clean_no_outliers = df_clean[~outlier_mask].copy()
    
    # Final data cleaning: remove any remaining infinite or NaN values
    print(f"\n=== FINAL DATA CLEANING ===")
    print(f"Data before final cleaning: {len(df_clean_no_outliers)} observations")
    
    # Helper function for comprehensive data validation
    def validate_dataframe_quality(df, features, target_col='earnings_surprise_ratio'):
        """
        Comprehensive validation of DataFrame quality using pandas operations.
        
        Args:
            df: DataFrame to validate
            features: List of feature column names
            target_col: Target column name
            
        Returns:
            dict: Validation results and cleaned DataFrame
        """
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # Ensure features and target_col are present
        for col in features + [target_col]:
            if col not in df.columns:
                df[col] = np.nan
        # Check for infinite values using pandas operations
        # Defensive: ensure df, features, and target_col are not None before using []
        if df is None:
            df = pd.DataFrame()
        if features is None:
            features = []
        if target_col is None:
            target_col = ''
        # Ensure features and target_col are present
        for col in features + [target_col]:
            if col not in df.columns:
                df[col] = np.nan
        # Check for infinite values using pandas operations
        if not features or not target_col or df.empty:
            inf_check = pd.Series([False], index=[target_col])
        else:
            inf_check = df[features + [target_col]].isin([np.inf, -np.inf]).any()
        if isinstance(inf_check, (bool, np.bool_)):
            inf_check = pd.Series([inf_check]*len(features+[target_col]), index=features+[target_col])
        elif isinstance(inf_check, np.ndarray):
            inf_check = pd.Series(inf_check, index=df[features + [target_col]].columns)
        elif inf_check is None:
            inf_check = pd.Series([False]*len(features+[target_col]), index=features+[target_col])
        else:
            try:
                inf_check = pd.Series(inf_check, index=df[features + [target_col]].columns)
            except Exception:
                inf_check = pd.Series([False]*len(features+[target_col]), index=features+[target_col])
        features_with_inf = inf_check[inf_check].index.tolist() if isinstance(inf_check, pd.Series) else []
        # Check for NaN values using pandas operations
        # Defensive: ensure df, features, and target_col are not None before using []
        if df is None:
            df = pd.DataFrame()
        if features is None:
            features = []
        if target_col is None:
            target_col = ''
        # Ensure features and target_col are present
        for col in features + [target_col]:
            if col not in df.columns:
                df[col] = np.nan
        # Check for NaN values using pandas operations
        if not features or not target_col or df.empty:
            nan_check = pd.Series([False], index=[target_col])
        else:
            nan_check = df[features + [target_col]].isna().any()
        if isinstance(nan_check, (bool, np.bool_)):
            nan_check = pd.Series([nan_check]*len(features+[target_col]), index=features+[target_col])
        elif isinstance(nan_check, np.ndarray):
            nan_check = pd.Series(nan_check, index=df[features + [target_col]].columns)
        elif nan_check is None:
            nan_check = pd.Series([False]*len(features+[target_col]), index=features+[target_col])
        else:
            try:
                nan_check = pd.Series(nan_check, index=df[features + [target_col]].columns)
            except Exception:
                nan_check = pd.Series([False]*len(features+[target_col]), index=features+[target_col])
        features_with_nan = nan_check[nan_check].index.tolist() if isinstance(nan_check, pd.Series) else []
        # Print warnings
        for feature in features_with_inf:
            print(f"  WARNING: Found infinite values in '{feature}'")
        for feature in features_with_nan:
            print(f"  WARNING: Found NaN values in '{feature}'")
        # Clean the data
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(0)
        return {
            'features_with_inf': features_with_inf,
            'features_with_nan': features_with_nan,
            'cleaned_df': df_clean,
            'rows_removed': len(df) - len(df_clean)
        }
    
    # Use helper function for validation
    validation_results = validate_dataframe_quality(df_clean_no_outliers, features)
    df_clean_final = validation_results['cleaned_df'] if validation_results and 'cleaned_df' in validation_results and validation_results['cleaned_df'] is not None else pd.DataFrame()
    print(f"  Removed {validation_results['rows_removed']} rows with invalid data" if validation_results and 'rows_removed' in validation_results else "  Removed unknown rows with invalid data")
    print(f"Data after final cleaning: {len(df_clean_final)} observations (removed {len(df_clean_no_outliers) - len(df_clean_final) if isinstance(df_clean_no_outliers, (pd.DataFrame, list)) and isinstance(df_clean_final, (pd.DataFrame, list)) else 'unknown'} additional rows)")
    if not isinstance(df_clean_final, pd.DataFrame) or len(df_clean_final) == 0:
        print("ERROR: No valid data remaining after cleaning. Cannot proceed with regression.")
        return
    
    # Create clean feature matrix and target vector
    X_clean = df_clean_final[features]
    y_clean = df_clean_final['earnings_surprise_ratio']
    
    # Final validation of data quality using pandas operations
    print(f"\n=== FINAL DATA VALIDATION ===")
    print(f"X_clean shape: {X_clean.shape}")
    print(f"y_clean shape: {y_clean.shape}")
    
    # Use pandas operations for efficient validation
    x_validation = {
        'has_inf': False,
        'has_nan': False
    }
    y_validation = {
        'has_inf': False,
        'has_nan': False
    }
    if isinstance(X_clean, pd.DataFrame):
        x_validation['has_inf'] = bool(X_clean.isin([np.inf, -np.inf]).any().any())
        x_validation['has_nan'] = bool(X_clean.isna().any().any())
    if isinstance(y_clean, (pd.Series, pd.DataFrame)):
        y_validation['has_inf'] = bool(y_clean.isin([np.inf, -np.inf]).any()) if hasattr(y_clean, 'isin') else False
        y_validation['has_nan'] = bool(y_clean.isna().any()) if hasattr(y_clean, 'isna') else False
    
    print(f"X_clean has infinite values: {x_validation['has_inf']}")
    print(f"X_clean has NaN values: {x_validation['has_nan']}")
    print(f"y_clean has infinite values: {y_validation['has_inf']}")
    print(f"y_clean has NaN values: {y_validation['has_nan']}")
    
    # Feature statistics for validation
    for feature in features:
        print(f"{feature}: mean={X_clean[feature].mean():.6f}, std={X_clean[feature].std():.6f}, min={X_clean[feature].min():.6f}, max={X_clean[feature].max():.6f}")
    
    print(f"Target (earnings_surprise_ratio): mean={y_clean.mean():.6f}, std={y_clean.std():.6f}, min={y_clean.min():.6f}, max={y_clean.max():.6f}")
    
    # Add constant term
    X_clean_with_const = sm.add_constant(X_clean)
    
    print(f"\nData after outlier removal and final cleaning: {len(X_clean)} observations (removed {len(df_clean) - len(X_clean)} total outliers and invalid data)")
    
    # === OLS REGRESSION WITH INDUSTRY FIXED EFFECTS AND QUARTER CONTROLS ===
    print("\n=== OLS REGRESSION WITH INDUSTRY FIXED EFFECTS AND QUARTER CONTROLS ===")
    
    # Create formula for OLS with industry fixed effects and quarter controls
    # Set Software - Applications as the baseline (reference category)
    formula = f'earnings_surprise_ratio ~ cap_intensity + rnd_to_btd + leverage + revenue_growth + current_inventory_ratio + profitability + ppe_total_assets + acquisitions_intensity + goodwill_change_intensity + intangible_assets_change_intensity + total_dna + earnings_quality + quarter_number + year + C(sector_baseline, Treatment(reference="{baseline_sector}"))'
    print(f"OLS Formula: {formula}")
    
    # Fit OLS model with industry fixed effects and clustered standard errors by ticker ONLY
    model_with_fe = smf.ols(formula, data=df_clean_final).fit(cov_type='cluster', cov_kwds={'groups': df_clean_final['ticker']})
    # (No clustering or normalization by sector/industry anywhere in the code)
    
    print(f"OLS with Industry FE and Quarter Controls R-squared: {model_with_fe.rsquared:.4f}")
    print(f"OLS with Industry FE and Quarter Controls Adjusted R-squared: {model_with_fe.rsquared_adj:.4f}")
    
    # Also fit OLS without fixed effects for comparison
    ols_model = sm.OLS(y_clean, X_clean_with_const).fit(cov_type='HC3')
    print(f"OLS without Industry FE R-squared: {ols_model.rsquared:.4f}")
    print(f"OLS without Industry FE Adjusted R-squared: {ols_model.rsquared_adj:.4f}")

    print("\n=== REGRESSION RESULTS (WITH INDUSTRY FIXED EFFECTS AND QUARTER CONTROLS) ===")
    print(model_with_fe.summary())
    
    # Print industry fixed effects coefficients
    print("\n=== INDUSTRY FIXED EFFECTS ===")
    # Look for any parameter that contains sector_baseline (more flexible pattern)
    fe_coefficients = model_with_fe.params[model_with_fe.params.index.str.contains('sector_baseline')]
    if len(fe_coefficients) > 0:
        print(f"Industry Fixed Effects (relative to {baseline_sector} baseline):")
        for industry, coef in fe_coefficients.items():
            # Extract the sector name from the parameter name
            if '[' in industry and ']' in industry:
                industry_name = industry.split('[')[1].split(']')[0]
            else:
                industry_name = industry.replace('C(sector_baseline, Treatment(reference="' + baseline_sector + '"))[', '').replace(']', '')
            print(f"  {industry_name}: {coef:.6f}")
        print(f"  {baseline_sector}: 0.000000 (baseline/reference category)")
    else:
        print("No industry fixed effects found in the model.")
        print("Available parameters:", list(model_with_fe.params.index))
    
    # Perform comprehensive diagnostics for OLS model
    print("\n=== OLS REGRESSION DIAGNOSTICS ===")
    ols_diagnostics = robust_regression_diagnostics(X_clean, y_clean, ols_model, cv_folds=5)
    
    # Calculate and print VIF for ALL independent variables (including treatment variables)
    print("\n=== VARIANCE INFLATION FACTORS (VIF) ANALYSIS - ALL VARIABLES ===")
    print("VIF measures multicollinearity among all independent variables.")
    print("VIF > 10 indicates high multicollinearity, VIF > 5 indicates moderate multicollinearity.")
    print("-" * 80)
    
    # Calculate VIF for all features
    vif_results = calculate_vif(X_clean, feature_names=X_clean.columns.tolist())
    
    # Sort VIF results by value (highest first)
    sorted_vif = sorted(vif_results.items(), key=lambda x: x[1] if x[1] != float('inf') else 0, reverse=True)
    
    print("VIF Results (sorted by severity):")
    print(f"{'Feature':<25} {'VIF':<10} {'Severity':<15}")
    print("-" * 50)
    
    high_vif_features = []
    moderate_vif_features = []
    
    for feature, vif in sorted_vif:
        if vif == float('inf'):
            severity = "PERFECT COLLINEARITY"
            print(f"{feature:<25} {'∞':<10} {severity:<15}")
            high_vif_features.append(feature)
        elif vif > 10:
            severity = "HIGH"
            print(f"{feature:<25} {vif:<10.2f} {severity:<15}")
            high_vif_features.append(feature)
        elif vif > 5:
            severity = "MODERATE"
            print(f"{feature:<25} {vif:<10.2f} {severity:<15}")
            moderate_vif_features.append(feature)
        else:
            severity = "LOW"
            print(f"{feature:<25} {vif:<10.2f} {severity:<15}")
    
    print("-" * 50)
    
    # Summary and recommendations - only for control variables
    treatment_variables = ['cap_intensity', 'rnd_to_btd']
    
    # Filter warnings to only include control variables
    high_vif_controls = [f for f in high_vif_features if f not in treatment_variables]
    moderate_vif_controls = [f for f in moderate_vif_features if f not in treatment_variables]
    
    if high_vif_controls:
        print(f"\n⚠️  HIGH MULTICOLLINEARITY DETECTED IN CONTROL VARIABLES:")
        print(f"   Control variables with VIF > 10: {', '.join(high_vif_controls)}")
        print("   Recommendation: Consider removing one or more of these control variables")
        print("   or using regularization techniques (Ridge/Lasso regression)")
    
    if moderate_vif_controls:
        print(f"\n⚠️  MODERATE MULTICOLLINEARITY DETECTED IN CONTROL VARIABLES:")
        print(f"   Control variables with VIF > 5: {', '.join(moderate_vif_controls)}")
        print("   Recommendation: Monitor these control variables closely")
    
    # Check if any control variables have high/moderate VIF
    control_variables = [col for col in X_clean.columns if col not in treatment_variables]
    control_vif_values = {k: v for k, v in vif_results.items() if k in control_variables}
    high_control_vif = [k for k, v in control_vif_values.items() if v > 10]
    moderate_control_vif = [k for k, v in control_vif_values.items() if v > 5 and v <= 10]
    
    if not high_control_vif and not moderate_control_vif:
        print(f"\n✅ LOW MULTICOLLINEARITY IN CONTROL VARIABLES:")
        print("   All control variables have VIF < 5, indicating good model specification")
    
    # Note about treatment variables
    treatment_high_vif = [f for f in high_vif_features if f in treatment_variables]
    treatment_moderate_vif = [f for f in moderate_vif_features if f in treatment_variables]
    
    if treatment_high_vif or treatment_moderate_vif:
        print(f"\n📊 TREATMENT VARIABLES VIF STATUS:")
        print("   Note: High VIF in treatment variables (cap_intensity, rnd_to_btd) is expected")
        print("   and not concerning as these are the main variables of interest.")
        if treatment_high_vif:
            print(f"   Treatment variables with VIF > 10: {', '.join(treatment_high_vif)}")
        if treatment_moderate_vif:
            print(f"   Treatment variables with VIF > 5: {', '.join(treatment_moderate_vif)}")
    
    # Store VIF results for later use
    vif_summary = {
        'high_vif_features': high_vif_features,
        'moderate_vif_features': moderate_vif_features,
        'all_vif_values': vif_results,
        'max_vif': max(vif_results.values()) if vif_results else 0,
        'mean_vif': np.mean([v for v in vif_results.values() if v != float('inf')]) if vif_results else 0
    }
    
    # Calculate significant features (features with p-value < 0.05)
    significant_features = []
    if hasattr(model_with_fe, 'pvalues'):
        for feature, pvalue in model_with_fe.pvalues.items():
            if feature != 'Intercept' and pvalue < 0.05:
                significant_features.append(feature)
    
    print(f"\nSignificant features (p < 0.05): {len(significant_features)} out of {len([col for col in X_clean.columns if col != 'Intercept'])}")
    if significant_features:
        print(f"   Significant features: {', '.join(significant_features)}")
    else:
        print("   No features are statistically significant at p < 0.05")
    
    print(f"\nVIF Summary Statistics:")
    print(f"   Maximum VIF: {vif_summary['max_vif']:.2f}")
    print(f"   Mean VIF: {vif_summary['mean_vif']:.2f}")
    print(f"   Features with VIF > 10: {len(high_vif_features)}")
    print(f"   Features with VIF > 5: {len(moderate_vif_features)}")

    # Model Diagnostics
    # =================
    print("\n=== MODEL DIAGNOSTICS ===")

    # Calculate R-squared and adjusted R-squared
    r_squared = ols_model.rsquared
    adj_r_squared = ols_model.rsquared_adj

    print(f"R-squared: {r_squared:.4f}")
    print(f"Adjusted R-squared: {adj_r_squared:.4f}")


    # Risk Assessment
    # ===============
    print("\n=== RISK ASSESSMENT ===")

    # Identify companies with aggressive capitalization practices
    # (top 20% of capitalization intensity)
    df_clean['cap_aggressive'] = df_clean['cap_intensity'] > df_clean['cap_intensity'].quantile(0.8)

    # Get list of high-risk companies
    high_risk_companies = pd.Series(df_clean[df_clean['cap_aggressive']]['ticker']).unique()
    print(f"High-risk companies (aggressive capitalization): {list(high_risk_companies)}")

    # Calculate average metrics for high-risk vs low-risk companies
    risk_analysis = df_clean.groupby('cap_aggressive')[['earnings_surprise_ratio', 'revenue_growth', 'leverage']].mean()
    print("\nAverage metrics by risk category:")
    print(risk_analysis)

    # Implied Alpha and Options Trading Strategy section removed - moved to separate alpha script
    # --- Robust outlier filtering for regression/scatter and histogram plots ---
    # Apply comprehensive outlier filtering with helper function
    filtering_result = apply_robust_outlier_filtering(df_clean, X, y, include_fallback=True)
    X_clean = filtering_result['X_clean']
    y_clean = filtering_result['y_clean']
    
    print(f"[DIAGNOSTIC] After robust filtering: {filtering_result['rows_remaining']} rows remain for regression.")
    if filtering_result['fallback_used']:
        print(f"[DIAGNOSTIC] After fallback filtering: {filtering_result['rows_remaining']} rows remain for regression.")
    
    if X_clean.empty or y_clean.empty:
        print("[ERROR] Still no data for regression chart. Plotting placeholder chart.")
        plt.figure(figsize=(12, 8))
        plt.title("No data available for regression chart", fontsize=16)
        plt.text(0.5, 0.5, "No data available", fontsize=16, color='gray', ha='center', va='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        robust_savefig("rnd_to_btd_vs_earnings_surprise_v1.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Proceed to plot regression charts with filtered data
        fallback_suffix = " (Fallback)" if filtering_result['fallback_used'] else ""
        
        # Ensure X_clean and y_clean are properly aligned
        if len(X_clean) != len(y_clean):
            print(f"[WARNING] X_clean length ({len(X_clean)}) != y_clean length ({len(y_clean)}). Aligning...")
            common_index = X_clean.index.intersection(y_clean.index)
            X_clean = X_clean.loc[common_index]
            y_clean = y_clean.loc[common_index]
        
        if not X_clean.empty and not y_clean.empty:
            # R&D Deferral Ratio vs OCF Margin
            plt.figure(figsize=(12, 8))
            tickers = X_clean.index if hasattr(X_clean, 'index') else None
            if tickers is not None and 'ticker' in df_clean.columns:
                unique_tickers = df_clean['ticker'].unique()
                cmap = plt.get_cmap('nipy_spectral', len(unique_tickers))
                color_map = {ticker: cmap(i) for i, ticker in enumerate(unique_tickers)}
                colors = [color_map.get(t, (0.5, 0.5, 0.5, 1.0)) for t in df_clean.loc[X_clean.index, 'ticker']]
                plt.scatter(X_clean['rnd_to_btd'], y_clean, c=colors, alpha=0.7)
            else:
                plt.scatter(X_clean['rnd_to_btd'], y_clean, alpha=0.7)
            plt.title(f"Earnings Surprise Ratio vs R&D to Book-Tax Difference{fallback_suffix}", fontsize=16)
            plt.xlabel("R&D to Book-Tax Difference", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            robust_savefig("rnd_to_btd_vs_earnings_surprise_v1.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Warning: Not enough data for regression/scatter plots. Skipping charts.")

    print("[DIAGNOSTIC] Regression charts exported.")

    # Data Export
    # ===========
    print("\n=== DATA EXPORT ===")

    # Save the cleaned dataset to CSV
    output_filename = "s174_tcja_earnings_output.csv"
    df_clean.to_csv(output_filename, index=False)
    print(f"Regression data exported to {output_filename}")

    # Save time-series metrics to CSV
    if time_series_metrics:
        ts_df = pd.DataFrame(time_series_metrics).T
        ts_filename = "s174_tcja_earnings_time_series.csv"
        ts_df.to_csv(ts_filename)
        print(f"Time-series analysis exported to {ts_filename}")

    # Save diagnostic statistics to CSV
    print("\n=== EXPORTING DIAGNOSTIC STATISTICS ===")
    
    # Collect diagnostic statistics
    diagnostic_stats = {}
    
    # Model performance statistics
    diagnostic_stats['model_performance'] = {
        'r_squared': r_squared,
        'adjusted_r_squared': model_with_fe.rsquared_adj,
        'f_statistic': model_with_fe.fvalue,
        'overall_model_pvalue': model_with_fe.f_pvalue,  # Overall model p-value (F-test)
        'total_observations': len(df_clean_final),
        'total_companies': df_clean_final['ticker'].nunique(),
        'significant_features': len(significant_features)
    }
    
    # VIF statistics
    if 'vif_summary' in locals():
        diagnostic_stats['vif_analysis'] = {
            'max_vif': vif_summary['max_vif'],
            'mean_vif': vif_summary['mean_vif'],
            'high_vif_count': len(vif_summary['high_vif_features']),
            'moderate_vif_count': len(vif_summary['moderate_vif_features']),
            'high_vif_features': ', '.join(vif_summary['high_vif_features']),
            'moderate_vif_features': ', '.join(vif_summary['moderate_vif_features'])
        }
    
    # Cross-validation results
    if 'ols_diagnostics' in locals() and 'cross_validation' in ols_diagnostics:
        cv_results = ols_diagnostics['cross_validation']
        diagnostic_stats['cross_validation'] = {}
        
        # Extract CV results with correct key names
        cv_keys = {
            'cv_rmse_mean': 'mean_mse',  # MSE is used instead of RMSE
            'cv_rmse_std': 'std_mse',
            'cv_r2_mean': 'mean_r2',
            'cv_r2_std': 'std_r2',
            'cv_folds': 'cv_folds'  # This might not exist, will be handled
        }
        
        for target_key, source_key in cv_keys.items():
            if source_key in cv_results:
                diagnostic_stats['cross_validation'][target_key] = cv_results[source_key]
    
    # Residual statistics
    if 'ols_diagnostics' in locals() and 'residual_stats' in ols_diagnostics:
        residual_stats = ols_diagnostics['residual_stats']
        diagnostic_stats['residual_analysis'] = {}
        
        # Safely extract residual statistics with fallbacks
        residual_keys = {
            'mean_residual': ['mean', 'residual_mean'],
            'std_residual': ['std', 'residual_std'],
            'skewness': ['skewness', 'residual_skewness'],
            'kurtosis': ['kurtosis', 'residual_kurtosis'],
            'jarque_bera_statistic': ['jarque_bera_statistic', 'jb_statistic', 'jarque_bera_stat'],
            'jarque_bera_pvalue': ['jarque_bera_pvalue', 'jb_pvalue', 'jarque_bera_pval']
        }
        
        for target_key, possible_keys in residual_keys.items():
            value = None
            for key in possible_keys:
                if key in residual_stats:
                    value = residual_stats[key]
                    break
            if value is not None:
                diagnostic_stats['residual_analysis'][target_key] = value
    
    # Heteroskedasticity tests
    if 'ols_diagnostics' in locals() and 'heteroskedasticity' in ols_diagnostics:
        het_tests = ols_diagnostics['heteroskedasticity']
        diagnostic_stats['heteroskedasticity_tests'] = {
            'breusch_pagan_statistic': het_tests['breusch_pagan_statistic'],
            'breusch_pagan_pvalue': het_tests['breusch_pagan_pvalue'],
            'white_statistic': het_tests['white_statistic'],
            'white_pvalue': het_tests['white_pvalue']
        }
    
    # Feature importance (coefficients and VIF scores only - no individual p-values)
    if hasattr(model_with_fe, 'params'):
        coefficients = model_with_fe.params
        
        # Calculate VIF scores for each variable
        vif_scores = calculate_vif(X_clean, feature_names=X_clean.columns.tolist())
        
        feature_importance = {}
        for feature in coefficients.index:
            if feature != 'Intercept':
                feature_importance[f'{feature}_coefficient'] = coefficients[feature]
                feature_importance[f'{feature}_vif'] = vif_scores.get(feature, float('inf'))
        
        diagnostic_stats['feature_importance'] = feature_importance
    
    # Save diagnostic statistics to CSV (flipped format)
    diagnostic_filename = "s174_tcja_earnings_diagnostics.csv"
    
    # Flatten the nested dictionary for CSV export
    flattened_stats = {}
    for category, stats in diagnostic_stats.items():
        for key, value in stats.items():
            flattened_stats[f"{category}_{key}"] = value
    
    # Create DataFrame and transpose it to flip rows and columns
    diagnostic_df = pd.DataFrame([flattened_stats])
    diagnostic_df_flipped = diagnostic_df.transpose()
    diagnostic_df_flipped.columns = ['Value']
    diagnostic_df_flipped.index.name = 'Metric'
    
    diagnostic_df_flipped.to_csv(diagnostic_filename, index=True)
    print(f"Diagnostic statistics exported to {diagnostic_filename} (flipped format)")

    # Visualization
    # =============
    print("\n=== GENERATING PLOTS ===")

    # Set up the plotting style for better readability
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

    # Filter to rolling 3-year window before plotting
    df_plot = filter_to_rolling_years(df_clean, 'date', years=3)
    # Calculate date range from the filtered dataset (actual data used in analysis)
    date_range_str = get_date_range_str(df_plot)

    # Determine whether to generate chart based on model significance
    print("\n=== CHART GENERATION ===")
    chart_selection = determine_significant_variable(model_with_fe, features)
    
    if chart_selection['should_generate_chart']:
        selected_var = chart_selection['selected_variable']
        var_info = chart_selection['variable_info']
        model_pvalue = chart_selection['model_pvalue']
        
        print(f"✅ Chart generation proceeding...")
        print(f"   Overall Model p-value: {model_pvalue:.6f} (statistically significant)")
        print(f"   Independent variable: {var_info['display_name']}")
        
        # Generate chart title
        chart_title = f"Earnings Surprise Ratio vs R&D over Book-to-Tax Difference — Top Statistically Significant Outliers"
        
        # Generate chart filename
        chart_filename = f"s174_tcja_earnings_chart.png"
        
        print(f"[PLOT DIAGNOSTIC] About to plot {selected_var} chart. df_plot shape: {df_plot.shape if hasattr(df_plot, 'shape') else 'No shape'}")
        if selected_var in df_plot.columns and 'earnings_surprise_ratio' in df_plot.columns:
            print(f"[PLOT DIAGNOSTIC] {selected_var} range: {df_plot[selected_var].min():.4f} to {df_plot[selected_var].max():.4f}")
        
        # Create S&P 500 data for trend lines
        sp500_full_df = df_clean[[selected_var, 'earnings_surprise_ratio', 'ticker']].dropna() if selected_var in df_clean.columns and 'earnings_surprise_ratio' in df_clean.columns and 'ticker' in df_clean.columns else pd.DataFrame()
        
        # Create tech trend line data - ensure tech_tickers exists and is not empty
        if 'tech_tickers' in locals() and tech_tickers and not sp500_full_df.empty:
            sp500_tech_full_df = sp500_full_df[sp500_full_df['ticker'].isin(tech_tickers)]
        else:
            sp500_tech_full_df = pd.DataFrame()
            print(f'[TRENDLINE WARNING] tech_tickers not available or empty, tech trend line will be skipped')
        
        print(f'[TRENDLINE DIAGNOSTIC] {selected_var} sp500 shape:', sp500_full_df.shape, 'columns:', list(sp500_full_df.columns))
        print(f'[TRENDLINE DIAGNOSTIC] {selected_var} sp500 tech shape:', sp500_tech_full_df.shape, 'columns:', list(sp500_tech_full_df.columns))
        
        # Additional debugging for trend line data
        if not sp500_full_df.empty:
            print(f'[TRENDLINE DEBUG] S&P 500 data has {len(sp500_full_df)} observations')
            print(f'[TRENDLINE DEBUG] S&P 500 data range: {selected_var} = {sp500_full_df[selected_var].min():.2f} to {sp500_full_df[selected_var].max():.2f}')
            print(f'[TRENDLINE DEBUG] S&P 500 data range: earnings_surprise_ratio = {sp500_full_df["earnings_surprise_ratio"].min():.2f} to {sp500_full_df["earnings_surprise_ratio"].max():.2f}')
        else:
            print(f'[TRENDLINE WARNING] S&P 500 data is empty - no trend line will be drawn')
        
        if not sp500_tech_full_df.empty:
            print(f'[TRENDLINE DEBUG] S&P 500 Tech data has {len(sp500_tech_full_df)} observations')
            print(f'[TRENDLINE DEBUG] S&P 500 Tech data range: {selected_var} = {sp500_tech_full_df[selected_var].min():.2f} to {sp500_tech_full_df[selected_var].max():.2f}')
            print(f'[TRENDLINE DEBUG] S&P 500 Tech data range: earnings_surprise_ratio = {sp500_tech_full_df["earnings_surprise_ratio"].min():.2f} to {sp500_tech_full_df["earnings_surprise_ratio"].max():.2f}')
        else:
            print(f'[TRENDLINE WARNING] S&P 500 Tech data is empty - no tech trend line will be drawn')
        
        plot_regression_chart(
            df_plot if selected_var in df_plot.columns else pd.DataFrame(),
            x_col=selected_var,
            y_col='earnings_surprise_ratio',
            chart_filename=chart_filename,
            title=chart_title,
            subtitle=None,  # Subtitle handled in plot_regression_chart
            date_range_str=date_range_str,
            sp500_df=sp500_full_df,  # S&P500 data for trend lines
            sp500_tech_df=sp500_tech_full_df,  # S&P500 tech data for trend lines
            variable_info=var_info  # Pass variable info for dynamic labels
        )
        print(f"[DIAGNOSTIC] {chart_filename} exported.")
    else:
        print(f"❌ Chart generation skipped: {chart_selection['reason']}")
        print(f"   Overall Model p-value: {chart_selection['model_pvalue']:.6f} (not statistically significant)")
        print(f"   No chart will be generated")

    print("All requested plots saved successfully!")

    # Summary Report
    # ==============
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total companies analyzed: {len(tickers)}")
    print(f"Total observations: {len(df_clean)}")
    print(f"Time period: {date_range_str}")
    print(f"Model R-squared: {r_squared:.4f}")
    print(f"Significant features: {len(significant_features)} out of {len(['rnd_to_btd', 'leverage', 'revenue_growth', 'current_inventory_ratio', 'profitability', 'ppe_total_assets', 'acquisitions_intensity', 'goodwill_change_intensity', 'intangible_assets_change_intensity', 'total_dna', 'log_market_cap', 'log_total_assets', 'earnings_quality', 'quarter_number'])}")
    print(f"High-risk companies: {len(high_risk_companies)}")
    
    if 'market_data' in locals() and 'sp500' in market_data:
        print(f"\nMarket Context:")
        print(f"S&P 500: ${market_data['sp500']['price']:,.2f} ({market_data['sp500']['changePercent']:+.2f}%)")
    else:
        print("[MARKET CONTEXT WARNING] S&P 500 data not available in market_data.")

    print("="*60)
    print("Analysis complete! Check the generated CSV files and plots for insights.")

    # At the END of main(), after all feature engineering and after df_clean is defined, insert:
    
    # Define the missing variables for plotting
    df_companies = ensure_pd_type(df_companies, 'dataframe')
    sp500_companies = ensure_pd_type(df_companies.copy(), 'dataframe')  # All S&P 500 companies
    # Ensure tech_tickers is a list and not an ndarray
    if not isinstance(tech_tickers, list):
        tech_tickers = list(tech_tickers)
    sp500_tech_companies = ensure_pd_type(df_companies[df_companies['ticker'].isin(tech_tickers)].copy(), 'dataframe')  # Tech companies only


    # === FINAL VALUE RANGE DIAGNOSTICS ===
    print("\n" + "="*80)
    print("FINAL VALUE RANGE DIAGNOSTICS")
    print("="*80)
    print("Checking if financial ratios are in reasonable ranges...")
    
    # Check cap_intensity (intangible assets / R&D expenses)
    if 'cap_intensity' in df_clean.columns:
        cap_intensity_stats = df_clean['cap_intensity'].describe()
        print(f"\ncap_intensity statistics:")
        print(f"  Count: {cap_intensity_stats['count']}")
        print(f"  Mean: {cap_intensity_stats['mean']:.2f}")
        print(f"  Std: {cap_intensity_stats['std']:.2f}")
        print(f"  Min: {cap_intensity_stats['min']:.2f}")
        print(f"  Max: {cap_intensity_stats['max']:.2f}")
        print(f"  25%: {cap_intensity_stats['25%']:.2f}")
        print(f"  50%: {cap_intensity_stats['50%']:.2f}")
        print(f"  75%: {cap_intensity_stats['75%']:.2f}")
        
        # Check for extreme values
        extreme_cap = df_clean[df_clean['cap_intensity'] > 100]
        if len(extreme_cap) > 0:
            print(f"  WARNING: {len(extreme_cap)} observations have cap_intensity > 100")
            print(f"  These may be causing filtering issues. Sample extreme values:")
            if not isinstance(extreme_cap, pd.DataFrame):
                extreme_cap = pd.DataFrame(extreme_cap)
            print(extreme_cap[['ticker', 'date', 'cap_intensity']].head())
    
    # Check ocf_margin (operating cash flow / net income)
    if 'ocf_margin' in df_clean.columns:
        ocf_margin_stats = df_clean['ocf_margin'].describe()
        print(f"\nocf_margin statistics:")
        print(f"  Count: {ocf_margin_stats['count']}")
        print(f"  Mean: {ocf_margin_stats['mean']:.2f}")
        print(f"  Std: {ocf_margin_stats['std']:.2f}")
        print(f"  Min: {ocf_margin_stats['min']:.2f}")
        print(f"  Max: {ocf_margin_stats['max']:.2f}")
        print(f"  25%: {ocf_margin_stats['25%']:.2f}")
        print(f"  50%: {ocf_margin_stats['50%']:.2f}")
        print(f"  75%: {ocf_margin_stats['75%']:.2f}")
        
        # Check for extreme values
        extreme_ocf = df_clean[df_clean['ocf_margin'] > 10]
        if len(extreme_ocf) > 0:
            print(f"  WARNING: {len(extreme_ocf)} observations have ocf_margin > 10")
            print(f"  These may be causing filtering issues. Sample extreme values:")
            if not isinstance(extreme_ocf, pd.DataFrame):
                extreme_ocf = pd.DataFrame(extreme_ocf)
            print(extreme_ocf[['ticker', 'date', 'ocf_margin']].head())
    

    
    # Check rnd_to_btd (R&D expenses / Book-to-Tax Difference)
    if 'rnd_to_btd' in df_clean.columns:
        rnd_to_btd_stats = df_clean['rnd_to_btd'].describe()
        print(f"\nrnd_to_btd statistics:")
        print(f"  Count: {rnd_to_btd_stats['count']}")
        print(f"  Mean: {rnd_to_btd_stats['mean']:.4f}")
        print(f"  Std: {rnd_to_btd_stats['std']:.4f}")
        print(f"  Min: {rnd_to_btd_stats['min']:.4f}")
        print(f"  Max: {rnd_to_btd_stats['max']:.4f}")
        print(f"  25%: {rnd_to_btd_stats['25%']:.4f}")
        print(f"  50%: {rnd_to_btd_stats['50%']:.4f}")
        print(f"  75%: {rnd_to_btd_stats['75%']:.4f}")
        
        # Check for extreme values
        invalid_rnd_btd = df_clean[(df_clean['rnd_to_btd'].abs() > 100) | df_clean['rnd_to_btd'].isna()]
        if len(invalid_rnd_btd) > 0:
            print(f"  WARNING: {len(invalid_rnd_btd)} observations have rnd_to_btd with extreme values (>100) or NaN")
            print(f"  These may be causing filtering issues. Sample invalid values:")
            if not isinstance(invalid_rnd_btd, pd.DataFrame):
                invalid_rnd_btd = pd.DataFrame(invalid_rnd_btd)
            print(invalid_rnd_btd[['ticker', 'date', 'rnd_to_btd']].head())
    
    print("\n=== END FINAL VALUE RANGE DIAGNOSTICS ===")
    print("="*80)

    # --- FINAL DIAGNOSTIC: Print at the very end of script execution ---
    print("\n" + "="*80)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Script completed at: {pd.Timestamp.now()}")
    print(f"Total tickers processed: {len(tickers_with_data) if 'tickers_with_data' in locals() else 'Unknown'}")
    print(f"Market data successfully fetched: {len(market_data_batch) if 'market_data_batch' in locals() else 'Unknown'}")
    if 'quotes_data' in locals():
        print(f"Quotes data type: {type(quotes_data)}")
        print(f"Quotes data shape: {quotes_data.shape if hasattr(quotes_data, 'shape') else 'No shape'}")
        if hasattr(quotes_data, 'shape') and quotes_data.shape[0] > 0:
            print(f"Sample quotes data: {quotes_data.iloc[0].to_dict()}")
    # All trading signal diagnostics removed

    # Print critical warnings summary at the very end
    print_critical_warnings_summary()


    # At the very end of main(), print cache status summary
    print("\n================ CACHE STATUS SUMMARY ================")
    for fname, exists, expired in cache_statuses:
        status = "EXPIRED/MISSING" if not exists or expired else "OK"
        print(f"{fname}: {status}")
    print("====================================================\n")

    # At the top of main(), add:
    chart_diagnostics = []
    # ...
    # At the very end of main(), after all other print statements:
    print("\n=== CHART EXPORT DIAGNOSTICS ===")
    for msg in chart_diagnostics:
        print(msg)
    
    # FINAL VIF ANALYSIS - Print at the very end of script execution
    print("\n" + "="*80)
    print("FINAL VARIANCE INFLATION FACTORS (VIF) ANALYSIS - ALL VARIABLES")
    print("="*80)
    print("This analysis is performed at the end of script execution to assess multicollinearity.")
    print("VIF measures how much the variance of a coefficient is inflated due to multicollinearity.")
    print("VIF > 10 indicates high multicollinearity, VIF > 5 indicates moderate multicollinearity.")
    print("-" * 80)
    
    try:
        # Calculate VIF for all variables in the final cleaned dataset
        if 'X_clean' in locals() and X_clean is not None and not X_clean.empty:
            final_vif_results = calculate_vif(X_clean, feature_names=X_clean.columns.tolist())
            
            # Sort VIF results by value (highest first)
            sorted_final_vif = sorted(final_vif_results.items(), key=lambda x: x[1] if x[1] != float('inf') else 0, reverse=True)
            
            print("FINAL VIF Results (sorted by severity):")
            print(f"{'Feature':<25} {'VIF':<10} {'Severity':<15} {'Interpretation':<30}")
            print("-" * 80)
            
            final_high_vif_features = []
            final_moderate_vif_features = []
            
            for feature, vif in sorted_final_vif:
                if vif == float('inf'):
                    severity = "PERFECT COLLINEARITY"
                    interpretation = "Remove one feature"
                    print(f"{feature:<25} {'∞':<10} {severity:<15} {interpretation:<30}")
                    final_high_vif_features.append(feature)
                elif vif > 10:
                    severity = "HIGH"
                    interpretation = "Consider removal"
                    print(f"{feature:<25} {vif:<10.2f} {severity:<15} {interpretation:<30}")
                    final_high_vif_features.append(feature)
                elif vif > 5:
                    severity = "MODERATE"
                    interpretation = "Monitor closely"
                    print(f"{feature:<25} {vif:<10.2f} {severity:<15} {interpretation:<30}")
                    final_moderate_vif_features.append(feature)
                else:
                    severity = "LOW"
                    interpretation = "Acceptable"
                    print(f"{feature:<25} {vif:<10.2f} {severity:<15} {interpretation:<30}")
            
            print("-" * 80)
            
            # Final summary and recommendations
            print(f"\nFINAL VIF SUMMARY:")
            print(f"   Total features analyzed: {len(final_vif_results)}")
            print(f"   Maximum VIF: {max(final_vif_results.values()) if final_vif_results else 0:.2f}")
            print(f"   Mean VIF: {np.mean([v for v in final_vif_results.values() if v != float('inf')]) if final_vif_results else 0:.2f}")
            print(f"   Features with VIF > 10: {len(final_high_vif_features)}")
            print(f"   Features with VIF > 5: {len(final_moderate_vif_features)}")
            
            # Final summary and recommendations - only for control variables
            treatment_variables = ['cap_intensity', 'rnd_to_btd']
            
            # Filter warnings to only include control variables
            final_high_vif_controls = [f for f in final_high_vif_features if f not in treatment_variables]
            final_moderate_vif_controls = [f for f in final_moderate_vif_features if f not in treatment_variables]
            
            if final_high_vif_controls:
                print(f"\n🚨 CRITICAL: HIGH MULTICOLLINEARITY DETECTED IN CONTROL VARIABLES:")
                print(f"   Control variables with VIF > 10: {', '.join(final_high_vif_controls)}")
                print("   RECOMMENDATION: Remove one or more of these control variables")
                print("   or use Ridge/Lasso regression to handle multicollinearity")
            
            elif final_moderate_vif_controls:
                print(f"\n⚠️  WARNING: MODERATE MULTICOLLINEARITY DETECTED IN CONTROL VARIABLES:")
                print(f"   Control variables with VIF > 5: {', '.join(final_moderate_vif_controls)}")
                print("   RECOMMENDATION: Monitor these control variables and consider regularization")
            
            else:
                print(f"\n✅ EXCELLENT: LOW MULTICOLLINEARITY IN CONTROL VARIABLES:")
                print("   All control variables have VIF < 5, indicating good model specification")
                print("   No multicollinearity concerns detected in control variables")
            
            # Note about treatment variables
            final_treatment_high_vif = [f for f in final_high_vif_features if f in treatment_variables]
            final_treatment_moderate_vif = [f for f in final_moderate_vif_features if f in treatment_variables]
            
            if final_treatment_high_vif or final_treatment_moderate_vif:
                print(f"\n📊 TREATMENT VARIABLES VIF STATUS:")
                print("   Note: High VIF in treatment variables (cap_intensity, rnd_to_btd) is expected")
                print("   and not concerning as these are the main variables of interest.")
                if final_treatment_high_vif:
                    print(f"   Treatment variables with VIF > 10: {', '.join(final_treatment_high_vif)}")
                if final_treatment_moderate_vif:
                    print(f"   Treatment variables with VIF > 5: {', '.join(final_treatment_moderate_vif)}")
            
            # Model quality assessment based on VIF (control variables only)
            print(f"\nMODEL QUALITY ASSESSMENT (CONTROL VARIABLES):")
            if len(final_high_vif_controls) > 0:
                print("   ❌ Model has significant multicollinearity issues in control variables")
                print("   Consider feature selection or regularization techniques for control variables")
            elif len(final_moderate_vif_controls) > 2:
                print("   ⚠️  Model has moderate multicollinearity concerns in control variables")
                print("   Consider monitoring or addressing high VIF control variables")
            else:
                print("   ✅ Model has good control variable independence")
                print("   Multicollinearity is not a significant concern in control variables")
                
                # VIF Analysis for Treatment Variables
                treatment_vars = ['rnd_to_btd', 'cap_intensity']
                print(f"\nTREATMENT VARIABLES VIF ANALYSIS DIAGNOSTICS:")
                print(f"Shape of X_clean[treatment_vars]: {X_clean[treatment_vars].shape}")
                print("Correlation matrix:")
                print(X_clean[treatment_vars].corr())
                print("First 10 rows:")
                print(X_clean[treatment_vars].head(10))
                print("Number of unique values:")
                for var in treatment_vars:
                    print(f"  {var}: {X_clean[var].nunique()} unique values")
                treatment_vif_results = calculate_vif(X_clean[treatment_vars], feature_names=treatment_vars)
                print(f"VIF Results for Treatment Variables:")
                for var, vif in treatment_vif_results.items():
                    print(f"   {var}: {vif:.3f}")
                
        else:
            print("❌ ERROR: Could not calculate VIF - X_clean not available")
            print("   This may indicate an issue with the data preprocessing pipeline")
            
    except Exception as e:
        print(f"❌ ERROR: Failed to calculate final VIF analysis: {e}")
        print("   This may indicate an issue with the VIF calculation function")
    
    print("="*80)
    print("END OF FINAL VIF ANALYSIS")
    print("="*80)

    # ... existing code ...
    print("[DIAGNOSTIC] Generating robust regression chart for rnd_to_btd...")
    # Prepare full S&P500 and S&P500 tech DataFrames for the chart
    sp500_full_df_rnd = df_clean[['rnd_to_btd', 'earnings_surprise_ratio', 'ticker']].dropna() if 'rnd_to_btd' in df_clean.columns and 'earnings_surprise_ratio' in df_clean.columns and 'ticker' in df_clean.columns else pd.DataFrame()
    sp500_tech_full_df_rnd = sp500_full_df_rnd[sp500_full_df_rnd['ticker'].isin(tech_tickers)] if not sp500_full_df_rnd.empty else pd.DataFrame()
    print('[TRENDLINE DIAGNOSTIC] rnd_to_btd S&P500 shape:', sp500_full_df_rnd.shape, 'columns:', list(sp500_full_df_rnd.columns))
    print('[TRENDLINE DIAGNOSTIC] rnd_to_btd S&P500 tech shape:', sp500_tech_full_df_rnd.shape, 'columns:', list(sp500_tech_full_df_rnd.columns))

    # Chart generation is handled above based on model significance
    # ... existing code ...

# Helper function to robustly convert any object to a DataFrame
def ensure_pd_type(obj, typ='dataframe'):
    """
    Convert any object to a pandas DataFrame or Series, handling all edge cases.
    typ: 'dataframe' or 'series'
    """
    import pandas as pd
    import numpy as np
    if typ == 'dataframe':
        if isinstance(obj, pd.DataFrame):
            return obj
        elif isinstance(obj, pd.Series):
            return obj.to_frame().T
        elif isinstance(obj, dict):
            return pd.DataFrame([obj])
        elif isinstance(obj, (list, np.ndarray)):
            if len(obj) == 0:
                return pd.DataFrame()
            if isinstance(obj[0], dict):
                return pd.DataFrame(obj)
            return pd.DataFrame(obj)
        else:
            return pd.DataFrame([obj])
    elif typ == 'series':
        if isinstance(obj, pd.Series):
            return obj
        elif isinstance(obj, pd.DataFrame):
            if obj.shape[1] == 1:
                return obj.iloc[:, 0]
            else:
                return obj.iloc[:, 0]  # fallback to first column
        elif isinstance(obj, (list, np.ndarray)):
            return pd.Series(obj)
        else:
            return pd.Series([obj])
    else:
        raise ValueError("typ must be 'dataframe' or 'series'")

def format_pval(p):
    return f"{p:.6f}" if p >= 1e-6 else f"{p:.2e}"

def warn_anomalous_pval(feature, p_value, ticker=None):
    prefix = f"[{ticker}] " if ticker else ""
    if p_value == 1.0:
        print(f"  {prefix}WARNING: p-value for '{feature}' is exactly 1.0. This may indicate a degenerate regressor or data issue.")
    if p_value < 1e-12:
        print(f"  {prefix}WARNING: p-value for '{feature}' is extremely low ({p_value:.2e}). This may indicate perfect collinearity or a data issue.")
    if p_value == 0.0 or p_value < 1e-300:
        print(f"  {prefix}WARNING: p-value for '{feature}' is IMPOSSIBLY LOW ({p_value}). This is likely a numerical or data issue, not a real statistical result.")


# 1. Define robust_outlier_mask at the top level (after imports)
def robust_outlier_mask(df, col, lower=0.01, upper=0.99):
    q_low = df[col].quantile(lower)
    q_high = df[col].quantile(upper)
    return (df[col] >= q_low) & (df[col] <= q_high)

# 2. Remove any top-level code that references df_clean, X, y, or df_companies.
# 3. Move all robust date range and outlier filtering logic inside main(), after df_clean and other variables are defined, and before any plotting.
# 4. All plotting and chart logic that uses these variables should be inside main().
# (No code outside main() should reference df_clean, X, y, or df_companies.)

def robust_outlier_filter_2d(df, x_col, y_col, z_thresh=3, lower=0.01, upper=0.99, min_points=10):
    print(f"\n[Diagnostics] Before filtering: {len(df)} rows")
    print(f"  {x_col} min/max: {df[x_col].min():.2f}/{df[x_col].max():.2f}")
    print(f"  {y_col} min/max: {df[y_col].min():.2f}/{df[y_col].max():.2f}")
    # Remove extreme values first
    df = df[(df[y_col] > -100) & (df[y_col] < 100)]
    # Percentile filter
    x_low, x_high = df[x_col].quantile([lower, upper])
    y_low, y_high = df[y_col].quantile([lower, upper])
    mask = (
        (df[x_col] >= x_low) & (df[x_col] <= x_high) &
        (df[y_col] >= y_low) & (df[y_col] <= y_high)
    )
    df_filt = df[mask].copy()
    # Z-score filter
    z_x = scipy.stats.zscore(df_filt[x_col].fillna(0))
    z_y = scipy.stats.zscore(df_filt[y_col].fillna(0))
    mask_z = (abs(z_x) < z_thresh) & (abs(z_y) < z_thresh)
    df_final = df_filt[mask_z].copy()
    # If too few points, relax filter
    if len(df_final['ticker'].unique()) < min_points:
        print(f"[Diagnostics] Too few tickers after filtering ({len(df_final['ticker'].unique())}), relaxing filter...")
        df_final = df[(df[y_col] > -100) & (df[y_col] < 100)].copy()
    print(f"[Diagnostics] After filtering: {len(df_final)} rows")
    print(f"  {x_col} min/max: {df_final[x_col].min():.2f}/{df_final[x_col].max():.2f}")
    print(f"  {y_col} min/max: {df_final[y_col].min():.2f}/{df_final[y_col].max():.2f}")
    print(f"  Rows removed: {len(df) - len(df_final)}")
    print(f"  Tickers plotted: {df_final['ticker'].unique().tolist()}")
    for t in df_final['ticker'].unique():
        print(f"    {t}: {len(df_final[df_final['ticker']==t])} points")
    return df_final

def get_quarter_year(date_str):
    # Expects date_str in 'YYYY-MM-DD' format
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    quarter = (dt.month - 1) // 3 + 1
    return f"Q{quarter} {dt.year}", dt.year, quarter

def get_date_range_str(df, date_col='date'):
    if date_col not in df.columns or df.empty:
        return ""
    dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
    if dates.empty:
        return ""
    
    # Calculate the 3-year rolling window based on the end date
    # This matches the logic in filter_to_rolling_years
    max_date = dates.max()
    end_year = max_date.year
    end_quarter = (max_date.month - 1) // 3 + 1
    
    # Start date is 3 years before the end date, same quarter
    start_year = end_year - 3
    start_quarter = end_quarter
    
    return f"Q{start_quarter} {start_year} to Q{end_quarter} {end_year}"

# Utility: Filter DataFrame to rolling N years

def filter_to_rolling_years(df, date_col='date', years=3):
    """
    Filter DataFrame to the last N years based on the maximum date in the dataset.
    Adjusted for lookback variables: ensures at least 3 extra quarters before the window for rolling calculations,
    and ends the window one quarter before the most recent for lookahead calculations.
    """
    if date_col not in df.columns or df.empty:
        print(f"[WARNING] Cannot filter {date_col} - column not found or DataFrame empty")
        return df
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        valid_dates = df_copy[date_col].notna()
        if not valid_dates.any():
            print(f"[WARNING] No valid dates found in {date_col}")
            return df
        df_copy = df_copy[valid_dates]
        unique_dates = sorted(df_copy[date_col].unique())
        if len(unique_dates) < 7:  # Need at least 7 for 4q rolling + lookback
            print(f"[WARNING] Not enough periods for lookback variables. Only {len(unique_dates)} periods available.")
            return df_copy
        # End date is the second-to-last period (for lookahead)
        end_date = unique_dates[-2]
        # Start date is 3 quarters before the window start
        start_date = end_date - pd.DateOffset(years=years) - pd.DateOffset(months=9)
        # Find the first date >= start_date
        for i, date in enumerate(unique_dates):
            if date >= start_date:
                if i > 2:  # Ensure at least 3 extra quarters before window
                    start_date = unique_dates[i-3]
                else:
                    start_date = unique_dates[0]
                break
        filtered_df = df_copy[(df_copy[date_col] >= start_date) & (df_copy[date_col] <= end_date)]
        return filtered_df
    except Exception as e:
        print(f"[ERROR] Error in filter_to_rolling_years: {e}")
        return df

def validate_financial_metric(value, metric_name, ticker, min_val=None, max_val=None, default_val=None):
    """
    Validate financial metrics and return cleaned values.
    
    Args:
        value: The raw metric value
        metric_name: Name of the metric for logging
        ticker: Stock ticker for logging
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        default_val: Default value if validation fails
    
    Returns:
        tuple: (cleaned_value, is_valid, warning_message)
    """
    if value is None:
        return default_val, False, f"Missing {metric_name} for {ticker}"
    
    try:
        value = float(value)
    except (ValueError, TypeError):
        return default_val, False, f"Invalid {metric_name} format for {ticker}: {value}"
    
    warning_msg = None
    
    # Check bounds
    if min_val is not None and value < min_val:
        warning_msg = f"{metric_name} for {ticker} ({value}) below minimum ({min_val})"
        if default_val is not None:
            value = default_val
    
    if max_val is not None and value > max_val:
        warning_msg = f"{metric_name} for {ticker} ({value}) above maximum ({max_val})"
        if default_val is not None:
            value = default_val
    
    return value, True, warning_msg

def fetch_company_profile(ticker, api_key, cache_file="profile_cache.json"):
    cache_expired = is_cache_expired(cache_file)
    if cache_expired and os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"[CACHE] Deleted expired cache: {cache_file}")
        except Exception as e:
            print(f"[CACHE ERROR] Could not delete expired cache {cache_file}: {e}")
    cache = load_cache(cache_file)
    if ticker in cache and not cache_expired:
        return cache[ticker]
    # If API key is None, we can't fetch from API - return None
    if api_key is None:
        print(f"[DEBUG] API key is None for {ticker} - cannot fetch profile from API")
        return None
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                profile = data[0]
                # Validate key metrics in profile data
                if 'beta' in profile:
                    beta, is_valid, warning = validate_financial_metric(
                        profile['beta'], 'beta', ticker, min_val=0, max_val=5, default_val=1.0
                    )
                    if warning:
                        print(f"[WARNING] {warning}")
                    profile['beta'] = beta
                if 'price' in profile:
                    price, is_valid, warning = validate_financial_metric(
                        profile['price'], 'price', ticker, min_val=0, max_val=10000, default_val=None
                    )
                    if warning:
                        print(f"[WARNING] {warning}")
                    profile['price'] = price
                cache[ticker] = profile
                save_cache(cache, cache_file)
                return profile
        else:
            print(f"[API ERROR] Failed to fetch profile for {ticker}: HTTP {response.status_code}")
            # Ensure no expired cache is used
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    print(f"[CACHE] Deleted expired cache after failed API call: {cache_file}")
                except Exception as e:
                    print(f"[CACHE ERROR] Could not delete expired cache {cache_file}: {e}")
    except Exception as e:
        print(f"Error fetching profile for {ticker}: {e}")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"[CACHE] Deleted expired cache after exception: {cache_file}")
            except Exception as e2:
                print(f"[CACHE ERROR] Could not delete expired cache {cache_file}: {e2}")
    return None

def cross_validate_regression_model(X, y, cv_folds=5, random_state=42):
    """
    Perform k-fold cross-validation on the regression model.
    
    Args:
        X: Feature matrix
        y: Target variable
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Cross-validation results including scores and coefficients
    """
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_scores = {
        'r2_scores': [],
        'adj_r2_scores': [],
        'mse_scores': [],
        'mae_scores': [],
        'coefficients': [],
        'intercepts': []
    }
    
    print(f"\n=== CROSS-VALIDATION RESULTS ({cv_folds}-fold) ===")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Add constant to training data
        X_train_with_const = sm.add_constant(X_train)
        X_test_with_const = sm.add_constant(X_test)
        
        # Fit model on training data
        model = sm.OLS(y_train, X_train_with_const).fit(cov_type='HC3')
        
        # Predict on test data
        y_pred = model.predict(X_test_with_const)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate adjusted R-squared
        n = len(y_test)
        p = len(X_train.columns)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Store results
        cv_scores['r2_scores'].append(r2)
        cv_scores['adj_r2_scores'].append(adj_r2)
        cv_scores['mse_scores'].append(mse)
        cv_scores['mae_scores'].append(mae)
        cv_scores['coefficients'].append(model.params[1:].to_dict())  # Exclude intercept
        cv_scores['intercepts'].append(model.params['const'])
        
        print(f"Fold {fold}: R² = {r2:.4f}, Adj R² = {adj_r2:.4f}, MSE = {mse:.6f}, MAE = {mae:.6f}")
    
    # Calculate summary statistics
    cv_summary = {
        'mean_r2': np.mean(cv_scores['r2_scores']),
        'std_r2': np.std(cv_scores['r2_scores']),
        'mean_adj_r2': np.mean(cv_scores['adj_r2_scores']),
        'std_adj_r2': np.std(cv_scores['adj_r2_scores']),
        'mean_mse': np.mean(cv_scores['mse_scores']),
        'std_mse': np.std(cv_scores['mse_scores']),
        'mean_mae': np.mean(cv_scores['mae_scores']),
        'std_mae': np.std(cv_scores['mae_scores']),
        'coefficient_stability': analyze_coefficient_stability(cv_scores['coefficients'])
    }
    
    print(f"\nCross-Validation Summary:")
    print(f"Mean R²: {cv_summary['mean_r2']:.4f} ± {cv_summary['std_r2']:.4f}")
    print(f"Mean Adj R²: {cv_summary['mean_adj_r2']:.4f} ± {cv_summary['std_adj_r2']:.4f}")
    print(f"Mean MSE: {cv_summary['mean_mse']:.6f} ± {cv_summary['std_mse']:.6f}")
    print(f"Mean MAE: {cv_summary['mean_mae']:.6f} ± {cv_summary['std_mae']:.6f}")
    
    return cv_scores, cv_summary

def analyze_coefficient_stability(coefficients_list):
    """
    Analyze the stability of coefficients across cross-validation folds.
    
    Args:
        coefficients_list: List of coefficient dictionaries from each fold
    
    Returns:
        dict: Coefficient stability metrics
    """
    
    # Convert to DataFrame for easier analysis
    coef_df = pd.DataFrame(coefficients_list)
    
    stability_metrics = {}
    for col in coef_df.columns:
        values = coef_df[col].dropna()
        if len(values) > 0:
            stability_metrics[col] = {
                'mean': values.mean(),
                'std': values.std(),
                'cv': values.std() / abs(values.mean()) if values.mean() != 0 else np.inf,
                'range': values.max() - values.min(),
                'sign_consistency': np.sum(np.sign(values) == np.sign(values.iloc[0])) / len(values)
            }
    
    print(f"\nCoefficient Stability Analysis:")
    for feature, metrics in stability_metrics.items():
        print(f"{feature}:")
        print(f"  Mean: {metrics['mean']:.6f}")
        print(f"  Std: {metrics['std']:.6f}")
        print(f"  CV: {metrics['cv']:.4f}")
        print(f"  Range: {metrics['range']:.6f}")
        print(f"  Sign Consistency: {metrics['sign_consistency']:.2f}")
    
    return stability_metrics

def improved_outlier_detection(df, features, target, method='isolation_forest'):
    """
    Improved outlier detection using multiple methods.
    
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        method: Outlier detection method ('isolation_forest', 'lof', 'robust_zscore', 'iqr')
    
    Returns:
        tuple: (outlier_mask, outlier_info)
    """
    
    X = df[features + [target]].copy()
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    
    if len(X) == 0:
        print("Warning: No valid data after removing infinite values")
        return pd.Series([False] * len(df)), {}
    
    outlier_info = {
        'method': method,
        'total_points': len(X),
        'outliers_detected': 0,
        'outlier_percentage': 0.0
    }
    
    if method == 'isolation_forest':
        # Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        outlier_mask = outlier_labels == -1
        
    elif method == 'lof':
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
        outlier_labels = lof.fit_predict(X)
        outlier_mask = outlier_labels == -1
        
    elif method == 'robust_zscore':
        # Robust Z-score using median and MAD
        outlier_mask = np.zeros(len(X), dtype=bool)
        for col in X.columns:
            median = X[col].median()
            mad = np.median(np.abs(X[col] - median))
            if mad > 0:
                z_scores = 0.6745 * (X[col] - median) / mad
                outlier_mask |= np.abs(z_scores) > 3.5
                
    elif method == 'iqr':
        # IQR method
        outlier_mask = np.zeros(len(X), dtype=bool)
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask |= (X[col] < lower_bound) | (X[col] > upper_bound)
    
    outlier_info['outliers_detected'] = np.sum(outlier_mask)
    outlier_info['outlier_percentage'] = np.sum(outlier_mask) / len(X) * 100
    
    print(f"\nOutlier Detection Results ({method}):")
    print(f"Total points: {outlier_info['total_points']}")
    print(f"Outliers detected: {outlier_info['outliers_detected']}")
    print(f"Outlier percentage: {outlier_info['outlier_percentage']:.2f}%")
    
    # Create mask for original DataFrame
    original_mask = pd.Series([False] * len(df), index=df.index)
    original_mask.loc[X.index] = outlier_mask
    
    return original_mask, outlier_info

def robust_ols_regression_diagnostics(X, y, ols_model, cv_folds=5):
    """
    Comprehensive ridge regression diagnostics including cross-validation.
    
    Args:
        X_scaled: Standardized feature matrix
        y: Target variable
        ridge_model: Fitted ridge regression model
        scaler: Fitted StandardScaler
        feature_names: List of feature names (should match X_scaled columns)
        cv_folds: Number of cross-validation folds
    
    Returns:
        dict: Comprehensive diagnostic results
    """
    
    # Get predictions and residuals using the pipeline
    y_pred = ridge_pipeline.predict(X_scaled)
    residuals = y - y_pred
    
    # Get the ridge model from the pipeline for alpha and coefficients
    ridge_model = ridge_pipeline.named_steps['ridgecv']
    
    # Basic model diagnostics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    # Calculate adjusted R-squared
    n = len(y)
    p = X_scaled.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    diagnostics = {
        'r_squared': r2,
        'adj_r_squared': adj_r2,
        'mse': mse,
        'rmse': rmse,
        'alpha': ridge_model.alpha_,
        'residual_stats': {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
    }
    
    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    diagnostics['normality_test'] = {
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_p,
        'is_normal': shapiro_p > 0.05
    }
    
    # Cross-validation
    cv_scores = cross_val_score(ridge_pipeline, X_scaled, y, cv=cv_folds, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    cv_r2 = cross_val_score(ridge_pipeline, X_scaled, y, cv=cv_folds, scoring='r2')
    
    diagnostics['cross_validation'] = {
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'cv_folds': cv_folds
    }
    
    # Coefficient analysis
    coefficients = ridge_model.coef_
    intercept = ridge_model.intercept_
    # Use feature_names for coefficient mapping, fallback if mismatch
    if len(feature_names) == len(coefficients):
        coef_dict = dict(zip(feature_names, coefficients))
        largest_idx = np.argmax(np.abs(coefficients))
        smallest_idx = np.argmin(np.abs(coefficients))
        largest_feature = feature_names[largest_idx]
        smallest_feature = feature_names[smallest_idx]
    diagnostics['coefficients'] = {
        'intercept': intercept,
        'feature_coefficients': coef_dict,
        'coefficient_magnitudes': np.abs(coefficients),
        'largest_coefficient': largest_feature,
        'smallest_coefficient': smallest_feature
    }
    
    # Print diagnostics
    print(f"\n=== RIDGE REGRESSION DIAGNOSTICS ===")
    print(f"Ridge Penalty (alpha): {ridge_model.alpha_:.6f}")
    print(f"R²: {diagnostics['r_squared']:.4f}")
    print(f"Adjusted R²: {diagnostics['adj_r_squared']:.4f}")
    print(f"MSE: {diagnostics['mse']:.6f}")
    print(f"RMSE: {diagnostics['rmse']:.6f}")
    
    print(f"\nResidual Statistics:")
    print(f"Mean: {diagnostics['residual_stats']['mean']:.6f}")
    print(f"Std: {diagnostics['residual_stats']['std']:.6f}")
    print(f"Skewness: {diagnostics['residual_stats']['skewness']:.4f}")
    print(f"Kurtosis: {diagnostics['residual_stats']['kurtosis']:.4f}")
    
    print(f"\nNormality Test (Shapiro-Wilk):")
    print(f"Statistic: {diagnostics['normality_test']['shapiro_statistic']:.4f}")
    print(f"P-value: {diagnostics['normality_test']['shapiro_pvalue']:.4f}")
    print(f"Normal residuals: {diagnostics['normality_test']['is_normal']}")
    
    print(f"\nCross-Validation Results:")
    print(f"CV RMSE: {diagnostics['cross_validation']['cv_rmse_mean']:.6f} (+/- {diagnostics['cross_validation']['cv_rmse_std'] * 2:.6f})")
    print(f"CV R²: {diagnostics['cross_validation']['cv_r2_mean']:.4f} (+/- {diagnostics['cross_validation']['cv_r2_std'] * 2:.4f})")
    
    print(f"\nCoefficient Analysis:")
    print(f"Intercept: {diagnostics['coefficients']['intercept']:.6f}")
    for feature, coef in diagnostics['coefficients']['feature_coefficients'].items():
        print(f"{feature}: {coef:.6f}")
    print(f"Largest coefficient: {diagnostics['coefficients']['largest_coefficient']}")
    print(f"Smallest coefficient: {diagnostics['coefficients']['smallest_coefficient']}")
    
    return diagnostics

def calculate_vif(X, feature_names=None):
    """
    Calculate Variance Inflation Factors (VIF) for all independent variables.
    
    Args:
        X: Feature matrix (DataFrame or numpy array)
        feature_names: List of feature names (optional)
    
    Returns:
        dict: VIF values for each feature
    """
    
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    vif_data = {}
    
    for feature in X.columns:
        # Regress the feature against all other features
        other_features = [col for col in X.columns if col != feature]
        
        if len(other_features) == 0:
            # Only one feature, VIF is undefined
            vif_data[feature] = float('inf')
            continue
            
        X_other = X[other_features]
        y_feature = X[feature]
        
        try:
            # Fit linear regression
            model = LinearRegression()
            model.fit(X_other, y_feature)
            
            # Calculate R-squared
            y_pred = model.predict(X_other)
            ss_res = np.sum((y_feature - y_pred) ** 2)
            ss_tot = np.sum((y_feature - np.mean(y_feature)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate VIF
            vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
            vif_data[feature] = vif
            
        except Exception as e:
            print(f"Error calculating VIF for {feature}: {e}")
            vif_data[feature] = float('inf')
    
    return vif_data

def robust_regression_diagnostics(X, y, model, cv_folds=5):
    """
    Comprehensive regression diagnostics including cross-validation.
    
    Args:
        X: Feature matrix
        y: Target variable
        model: Fitted regression model
        cv_folds: Number of cross-validation folds
    
    Returns:
        dict: Comprehensive diagnostic results
    """
    
    # Get residuals
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Basic model diagnostics
    diagnostics = {
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'model_pvalue': model.f_pvalue,  # Model p-value for overall significance
        'condition_number': np.linalg.cond(X),
        'residual_stats': {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
    }
    
    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    diagnostics['normality_test'] = {
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_p,
        'is_normal': shapiro_p > 0.05
    }
    
    # Heteroskedasticity test (Breusch-Pagan)
    try:
        bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, X)
        diagnostics['heteroskedasticity_test'] = {
            'breusch_pagan_statistic': bp_stat,
            'breusch_pagan_pvalue': bp_p,
            'is_heteroskedastic': bp_p < 0.05
        }
    except Exception:
        diagnostics['heteroskedasticity_test'] = {'error': 'Could not compute Breusch-Pagan test'}
    
    # Multicollinearity check
    vif_data = []
    for i, feature in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({'feature': feature, 'vif': vif})
    diagnostics['multicollinearity'] = vif_data
    
    # Cross-validation
    cv_scores, cv_summary = cross_validate_regression_model(X, y, cv_folds)
    diagnostics['cross_validation'] = cv_summary
    
    # Print diagnostics
    print(f"\n=== COMPREHENSIVE REGRESSION DIAGNOSTICS ===")
    print(f"R²: {diagnostics['r_squared']:.4f}")
    print(f"Adjusted R²: {diagnostics['adj_r_squared']:.4f}")
    print(f"F-statistic: {diagnostics['f_statistic']:.2f}")
    print(f"Overall Model p-value: {diagnostics['model_pvalue']:.6f}")
    
    # Check if model is statistically significant
    if diagnostics['model_pvalue'] < 0.05:
        print(f"✅ Model is statistically significant (p < 0.05)")
        print(f"   Chart will be generated")
    else:
        print(f"❌ Model is NOT statistically significant (p >= 0.05)")
        print(f"   Chart generation will be skipped due to lack of statistical significance")
    
    print(f"Condition number: {diagnostics['condition_number']:.2f}")
    
    print(f"\nResidual Statistics:")
    print(f"Mean: {diagnostics['residual_stats']['mean']:.6f}")
    print(f"Std: {diagnostics['residual_stats']['std']:.6f}")
    print(f"Skewness: {diagnostics['residual_stats']['skewness']:.4f}")
    print(f"Kurtosis: {diagnostics['residual_stats']['kurtosis']:.4f}")
    
    print(f"\nNormality Test (Shapiro-Wilk):")
    print(f"Statistic: {diagnostics['normality_test']['shapiro_statistic']:.4f}")
    print(f"P-value: {diagnostics['normality_test']['shapiro_pvalue']:.4f}")
    print(f"Normal residuals: {diagnostics['normality_test']['is_normal']}")
    
    if 'heteroskedasticity_test' in diagnostics and 'error' not in diagnostics['heteroskedasticity_test']:
        print(f"\nHeteroskedasticity Test (Breusch-Pagan):")
        print(f"Statistic: {diagnostics['heteroskedasticity_test']['breusch_pagan_statistic']:.4f}")
        print(f"P-value: {diagnostics['heteroskedasticity_test']['breusch_pagan_pvalue']:.4f}")
        print(f"Heteroskedastic: {diagnostics['heteroskedasticity_test']['is_heteroskedastic']}")
    
    print(f"\nMulticollinearity (VIF):")
    for vif_item in diagnostics['multicollinearity']:
        print(f"{vif_item['feature']}: {vif_item['vif']:.2f}")
    
    return diagnostics

# --- Regression Chart Generation (Top 10 Outliers, Full Title, Subtitle, Legend, R², Dot Size by Recency) ---
def plot_regression_chart(df, x_col, y_col, chart_filename, title, subtitle, date_range_str, sp500_df=None, sp500_tech_df=None, variable_info=None):
    print(f"[TRENDLINE DIAGNOSTIC] sp500_df: shape={getattr(sp500_df, 'shape', None)}, columns={getattr(sp500_df, 'columns', None)}")
    print(f"[TRENDLINE DIAGNOSTIC] sp500_tech_df: shape={getattr(sp500_tech_df, 'shape', None)}, columns={getattr(sp500_tech_df, 'columns', None)}")
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.text(0.5, 0.5, "No data available for this period", fontsize=16, color='gray', ha='center', va='center', transform=ax.transAxes)
        fig.suptitle(subtitle, fontsize=18, fontweight='bold', y=0.98)
        fig.tight_layout()
        robust_savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    # Use the date_range_str parameter passed from main function (which has correct rolling window logic)
    # Only calculate if date_range_str is empty or None
    if not date_range_str and 'date' in df.columns:
        min_date = pd.to_datetime(df['date']).min()
        max_date = pd.to_datetime(df['date']).max()
        start_year = min_date.year
        start_quarter = (min_date.month - 1) // 3 + 1
        end_year = max_date.year
        end_quarter = (max_date.month - 1) // 3 + 1
        date_range_str = f"Q{start_quarter} {start_year} to Q{end_quarter} {end_year}"
    x10, x90 = df[x_col].quantile([0.10, 0.90])
    y10, y90 = df[y_col].quantile([0.10, 0.90])
    mask = (df[x_col] >= x10) & (df[x_col] <= x90) & (df[y_col] >= y10) & (df[y_col] <= y90)
    df_filtered = df[mask].copy()
    X = df_filtered[[x_col]].values
    y = df_filtered[y_col].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    abs_resid = np.abs(residuals)
    r2 = model.score(X, y)
    if 'ticker' in df_filtered.columns:
        df_filtered['abs_resid'] = abs_resid
        top10_tickers = df_filtered.groupby('ticker')['abs_resid'].max().abs().sort_values(ascending=False).head(10).index.tolist()
        df_top10 = df[df['ticker'].isin(top10_tickers)].copy()
        df_top10['ticker'] = df_top10['ticker'].astype('category')
        roygbiv_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'magenta', 'cyan', 'brown']
        ticker_to_color = {ticker: roygbiv_colors[i % len(roygbiv_colors)] for i, ticker in enumerate(df_top10['ticker'].cat.categories)}
        colors = [ticker_to_color[ticker] for ticker in df_top10['ticker']]
    else:
        df_top10 = df_filtered.copy()
        colors = ['blue'] * len(df_top10)
    if 'date' in df_top10.columns:
        max_date = pd.to_datetime(df_top10['date']).max()
        min_date = pd.to_datetime(df_top10['date']).min()
        date_range = (max_date - min_date).days or 1
        recency = ((pd.to_datetime(df_top10['date']) - min_date).dt.days / date_range)
        bins = pd.cut(recency, bins=[-0.01, 0.25, 0.5, 0.75, 1.01], labels=[0.225, 0.55, 0.8, 1.0])
        sizes = bins.astype(float) * 960
    else:
        sizes = np.full(len(df_top10), 400)
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(df_top10[x_col], df_top10[y_col], s=sizes, c=colors, alpha=0.7, zorder=2)
    handles = []
    labels = []
    if 'ticker' in df_top10.columns:
        roygbiv_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'magenta', 'cyan', 'brown']
        for i, ticker in enumerate(df_top10['ticker'].cat.categories):
            color = roygbiv_colors[i % len(roygbiv_colors)]
            handles.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10))
            labels.append(ticker)
    trend_handles = []
    # Plot trend lines after scatter, with high zorder
    if sp500_df is not None and not sp500_df.empty and x_col in sp500_df.columns and y_col in sp500_df.columns:
        X_sp = sp500_df[[x_col]].values
        y_sp = sp500_df[y_col].values
        if len(X_sp) > 1:
            x_min, x_max = X_sp.min(), X_sp.max()
            x_vals = np.linspace(x_min, x_max, 100)
            model_sp = LinearRegression().fit(X_sp, y_sp)
            y_sp_vals = model_sp.predict(x_vals.reshape(-1, 1))
            h_sp = ax.plot(x_vals, y_sp_vals, color='blue', linestyle='--', linewidth=2, label='S&P 500 Trend', zorder=3)[0]
            trend_handles.append(h_sp)
    if sp500_tech_df is not None and not sp500_tech_df.empty and x_col in sp500_tech_df.columns and y_col in sp500_tech_df.columns:
        X_tech = sp500_tech_df[[x_col]].values
        y_tech = sp500_tech_df[y_col].values
        if len(X_tech) > 1:
            x_min_tech = x_min if 'x_min' in locals() else X_tech.min()
            x_max_tech = x_max if 'x_max' in locals() else X_tech.max()
            x_vals_tech = np.linspace(x_min_tech, x_max_tech, 100)
            model_tech = LinearRegression().fit(X_tech, y_tech)
            y_tech_vals = model_tech.predict(x_vals_tech.reshape(-1, 1))
            h_tech = ax.plot(x_vals_tech, y_tech_vals, color='purple', linestyle=':', linewidth=2, label='S&P 500 Tech Trend', zorder=3)[0]
            trend_handles.append(h_tech)
    # Title, labels - Use R&D over Book-to-Tax Difference for x-axis
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.98)
    ax.set_xlabel("R&D over Book-to-Tax Difference\n(R&D Expense / (Income Before Tax - Income Tax Expense))", fontsize=14)
    if y_col == 'earnings_surprise_ratio':
        ax.set_ylabel("Earnings Surprise Ratio (Actual/Estimated)", fontsize=14)
    else:
        ax.set_ylabel(y_col + " (4Q)" if "Margin" in y_col else y_col, fontsize=14)
    
    # Always display R² score prominently
    ax.text(0.02, 0.98, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=14, ha='left', va='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=5)
    plotted_x = df_top10[x_col]
    plotted_y = df_top10[y_col]
    x_range = plotted_x.max() - plotted_x.min()
    y_range = plotted_y.max() - plotted_y.min()
    x_margin = max(x_range * 0.1, 0.1) if x_range > 0 else 0.1
    y_margin = max(y_range * 0.1, 0.1) if y_range > 0 else 0.1
    x_min = plotted_x.min() - x_margin
    x_max = plotted_x.max() + x_margin
    ax.set_xlim(x_min, x_max)
    if y_col == 'earnings_surprise_ratio':
        y_abs_max = max(abs(plotted_y.min()), abs(plotted_y.max()))
        ax.set_ylim(-y_abs_max - y_margin, y_abs_max + y_margin)
    else:
        ax.set_ylim(plotted_y.min() - y_margin, plotted_y.max() + y_margin)
    combined_handles = handles + trend_handles
    combined_labels = labels + [h.get_label() for h in trend_handles]
    legend = None
    # --- Always plot trend lines across the full x-axis range ---
    xlim = ax.get_xlim()
    x_trend = np.linspace(xlim[0], xlim[1], 200)
    # S&P 500 trend line
    if sp500_df is not None and not sp500_df.empty and x_col in sp500_df.columns and y_col in sp500_df.columns:
        X_sp = sp500_df[[x_col]].values
        y_sp = sp500_df[y_col].values
        if len(X_sp) > 1:
            model_sp = LinearRegression().fit(X_sp, y_sp)
            y_sp_vals = model_sp.predict(x_trend.reshape(-1, 1))
            h_sp = ax.plot(x_trend, y_sp_vals, color='blue', linestyle='--', linewidth=2, label='S&P 500 Trend', zorder=3)[0]
            trend_handles.append(h_sp)
            print(f"[TRENDLINE DEBUG] S&P 500 trend line plotted with {len(X_sp)} points")
        else:
            print(f"[TRENDLINE DEBUG] S&P 500 trend line skipped - insufficient data points: {len(X_sp)}")
    else:
        print(f"[TRENDLINE DEBUG] S&P 500 trend line skipped - sp500_df is None or empty")
        if sp500_df is not None:
            print(f"[TRENDLINE DEBUG] sp500_df shape: {sp500_df.shape}, columns: {list(sp500_df.columns)}")
            print(f"[TRENDLINE DEBUG] Required columns: {x_col}, {y_col}")
    # S&P 500 Tech trend line
    if sp500_tech_df is not None and not sp500_tech_df.empty and x_col in sp500_tech_df.columns and y_col in sp500_tech_df.columns:
        X_tech = sp500_tech_df[[x_col]].values
        y_tech = sp500_tech_df[y_col].values
        if len(X_tech) > 1:
            model_tech = LinearRegression().fit(X_tech, y_tech)
            y_tech_vals = model_tech.predict(x_trend.reshape(-1, 1))
            h_tech = ax.plot(x_trend, y_tech_vals, color='purple', linestyle=':', linewidth=2, label='S&P 500 Tech Trend', zorder=3)[0]
            trend_handles.append(h_tech)
            print(f"[TRENDLINE DEBUG] S&P 500 Tech trend line plotted with {len(X_tech)} points")
        else:
            print(f"[TRENDLINE DEBUG] S&P 500 Tech trend line skipped - insufficient data points: {len(X_tech)}")
    else:
        print(f"[TRENDLINE DEBUG] S&P 500 Tech trend line skipped - sp500_tech_df is None or empty")
        if sp500_tech_df is not None:
            print(f"[TRENDLINE DEBUG] sp500_tech_df shape: {sp500_tech_df.shape}, columns: {list(sp500_tech_df.columns)}")
            print(f"[TRENDLINE DEBUG] Required columns: {x_col}, {y_col}")
    # Create legend only when there are handles
    legend = None
    if combined_handles:
        legend = ax.legend(combined_handles, combined_labels, title="Company", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., frameon=True, fontsize=12, title_fontsize=14, markerscale=1.0)
    
    # Place subtitle just below the legend, centered below the legend
    if legend is not None:
            # Use the legend's anchor position and get its actual rendered position
            fig.canvas.draw()
            
            # Get the legend's bounding box in display coordinates
            legend_box = legend.get_window_extent()
            
            # Convert to figure coordinates
            legend_box_fig = legend_box.transformed(fig.transFigure.inverted())
            
            # Use the legend's actual position with a hard-coded adjustment for visual effect
            # The adjustment compensates for the legend's internal padding and positioning
            text_centering_compensation = 0.105  # Increased compensation for main script
            legend_left_x = legend_box_fig.xmin - text_centering_compensation
            
            # Temporary debug output
            print(f"[DEBUG] Legend box: xmin={legend_box_fig.xmin:.3f}, xmax={legend_box_fig.xmax:.3f}")
            print(f"[DEBUG] Using left edge: {legend_left_x:.3f}")
            
            # Position subtitles below the legend with a small margin
            subtitle_y_start = legend_box_fig.ymin - 0.02  # Small margin below legend
            subtitle_line_spacing = 0.035  # Space between lines
            
            # Store subtitle positions for placement after tight_layout
            subtitle_positions = {
                'legend_left_x': legend_left_x,
                'subtitle_y_start': subtitle_y_start,
                'subtitle_line_spacing': subtitle_line_spacing
            }

    # Apply tight_layout first
    fig.tight_layout()
    
    # Now place subtitles after tight_layout to ensure they're not cut off
    if 'subtitle_positions' in locals():
        legend_left_x = subtitle_positions['legend_left_x']
        subtitle_y_start = subtitle_positions['subtitle_y_start']
        subtitle_line_spacing = subtitle_positions['subtitle_line_spacing']
        
        # Place subtitles with explicit color and zorder
        fig.text(legend_left_x, subtitle_y_start, date_range_str, fontsize=12, ha='left', va='top', color='black', zorder=10)
        fig.text(legend_left_x, subtitle_y_start - subtitle_line_spacing, "Rolling 4-Quarter Basis", fontsize=12, ha='left', va='top', color='black', zorder=10)
        fig.text(legend_left_x, subtitle_y_start - 2*subtitle_line_spacing, "Larger dot = more recent quarter", fontsize=12, ha='left', va='top', color='black', zorder=10)
    robust_savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def robust_savefig(filename, *args, **kwargs):
    """Save a figure and only print export message if file was actually updated."""
    try:
        # print(f"[DEBUG] Attempting to save figure to {filename} ...")
        mtime_before = os.path.getmtime(filename) if os.path.exists(filename) else None
    except Exception:
        mtime_before = None
    try:
        plt.savefig(filename, *args, **kwargs)
        plt.close()
        time.sleep(0.1)  # Ensure file system updates
        mtime_after = os.path.getmtime(filename) if os.path.exists(filename) else None
        if mtime_after and (mtime_before is None or mtime_after > mtime_before):
            print(f"[DIAGNOSTIC] {filename} exported. Size: {os.path.getsize(filename)} bytes")
        else:
            print(f"[WARNING] {filename} was NOT updated (no file change detected after savefig). Check for errors or empty data.")
    except Exception as e:
        print(f"[ERROR] Failed to export {filename}: {e}")
        # Force close the figure to prevent memory leaks
        plt.close('all')

def robust_save_cache(cache, cache_file):
    try:
        # print(f"[DEBUG] Attempting to save cache to {cache_file} ...")
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        time.sleep(0.1)
        if os.path.exists(cache_file):
            print(f"[DIAGNOSTIC] {cache_file} exported. Size: {os.path.getsize(cache_file)} bytes")
        else:
            print(f"[WARNING] {cache_file} was NOT created or updated.")
    except Exception as e:
        print(f"[ERROR] Failed to export {cache_file}: {e}")

def safe_get(df, i, col):
    try:
        if df is not None and hasattr(df, 'iloc') and hasattr(df, 'columns'):
            if 0 <= i < len(df) and col in df.columns:
                val = df.iloc[i][col]
                return val if not pd.isna(val) else np.nan
    except Exception:
        pass
    return np.nan

# --- GLOBAL FUNCTION: Rolling 4-Quarter Delta ---
def rolling_4q_delta(series, idx):
    """
    Calculate the difference between the value 4 quarters ago and the current value.
    Returns np.nan if not enough data or if input is not a Series.
    """
    if not isinstance(series, pd.Series) or len(series) < 4 or idx < 3:
        return np.nan
    window = series.iloc[idx-3:idx+1] if hasattr(series, 'iloc') else series[idx-3:idx+1]
    if len(window) == 4 and not any(pd.isna(window)):
        return window.iloc[-1] - window.iloc[0] if hasattr(window, 'iloc') else window[-1] - window[0]
    return np.nan

# --- GLOBAL FUNCTION: Rolling 4-Quarter Sum ---
def rolling_4q_sum(series, idx):
    """
    Calculate the sum over the current and previous 3 quarters.
    Returns np.nan if not enough data or if input is not a Series.
    """
    if not isinstance(series, pd.Series) or len(series) < 4 or idx < 3:
        return np.nan
    window = series.iloc[idx-3:idx+1] if hasattr(series, 'iloc') else series[idx-3:idx+1]
    if len(window) == 4 and not any(pd.isna(window)):
        return window.sum()
    return np.nan

# --- GLOBAL FUNCTION: Rolling 4-Quarter Average ---
def rolling_4q_avg(series, idx):
    """
    Calculate the average over the current and previous 3 quarters.
    Returns np.nan if not enough data or if input is not a Series.
    """
    if not isinstance(series, pd.Series) or len(series) < 4 or idx < 3:
        return np.nan
    window = series.iloc[idx-3:idx+1] if hasattr(series, 'iloc') else series[idx-3:idx+1]
    if len(window) == 4 and not any(pd.isna(window)):
        return window.mean()
    return np.nan

# Add this function after the existing functions, before main()

def determine_significant_variable(model_with_fe, features):
    """
    Determine whether to generate chart based on model significance.
    Always uses R&D over Book-to-Tax Difference as the independent variable.
    
    Args:
        model_with_fe: The fitted regression model
        features: List of feature names
    
    Returns:
        dict: Contains chart generation decision and variable info
    """
    if not hasattr(model_with_fe, 'params') or not hasattr(model_with_fe, 'pvalues'):
        return {'should_generate_chart': False, 'reason': 'Model parameters not available'}
    
    # Check if model is statistically significant using F-test p-value
    model_pvalue = model_with_fe.f_pvalue
    
    if model_pvalue >= 0.05:
        return {
            'should_generate_chart': False, 
            'reason': f'Model is not statistically significant (p-value: {model_pvalue:.4f} >= 0.05)',
            'model_pvalue': model_pvalue
        }
    
    # Always use R&D over BTD as the independent variable
    selected_var = 'rnd_to_btd'
    
    # Define variable info for R&D over BTD
    variable_info = {
        'rnd_to_btd': {
            'display_name': 'R&D over Book-to-Tax Difference',
            'formula': '(R&D Expense / (Income Before Tax - (Income Tax Expense / Effective Tax Rate)))',
            'title_suffix': 'R&D over BTD'
        }
    }
    
    return {
        'should_generate_chart': True,
        'selected_variable': selected_var,
        'variable_info': variable_info[selected_var],
        'model_pvalue': model_pvalue,
        'reason': f'Model is statistically significant (p-value: {model_pvalue:.4f} < 0.05)'
    }

if __name__ == "__main__":
    main()

  

   