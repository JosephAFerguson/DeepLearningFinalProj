import os
import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv # pip install python-dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("CMC_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Make sure you have a .env file with CMC_API_KEY.")

BASE_URL = "https://pro-api.coinmarketcap.com"
HEADERS = {
    'X-CMC_PRO_API_KEY': API_KEY,
    'Accept': 'application/json'
}

# --- 1. Get List of Coins ---
def get_top_coins(limit=50): # Increased limit to 50 to allow for filtering
    url = f"{BASE_URL}/v1/cryptocurrency/map"
    params = {
        'sort': 'cmc_rank',
        'limit': limit,
        'listing_status': 'active'
    }
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()['data']
    
    # --- FILTER OUT STABLECOINS ---
    # We remove any coin with 'usd' in the name or symbol, or explicit stablecoin tags
    # Note: A robust check would inspect the 'tags' field if available in this endpoint,
    # but checking symbol/name is a good heuristic for top coins.
    clean_list = []
    for coin in data:
        name = coin['name'].lower()
        symbol = coin['symbol'].lower()
        
        if 'usd' in name or 'usd' in symbol or 'tether' in name or 'dai' in name:
            print(f"Skipping Stablecoin: {coin['name']} ({coin['symbol']})")
            continue
            
        clean_list.append(coin)
        
    # Return only the top 'limit' (e.g. 30) after filtering
    return clean_list[:30]

# --- 2. Get Coin Metadata (for Graph) ---
def get_coin_metadata(coin_ids):
    url = f"{BASE_URL}/v2/cryptocurrency/info"
    ids_string = ",".join([str(c['id']) for c in coin_ids])
    params = {'id': ids_string}
    
    response = requests.get(url, headers=HEADERS, params=params)
    return response.json()['data']

# --- 3. Get Coin Price History (Micro Features) ---
def get_historical_prices(coin_id, days=365):
    url = f"{BASE_URL}/v1/cryptocurrency/quotes/historical"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'id': coin_id,
        'time_start': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'time_end': end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'interval': 'daily',
        'count': 10000 
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()
    
    if response.status_code != 200:
        print(f"  ! Failed ID {coin_id}: {data.get('status', {}).get('error_message')}")
        return []

    try:
        # Handle API response structure variations
        if 'quotes' in data['data']:
            quotes = data['data']['quotes']
        else:
            quotes = data['data'][str(coin_id)]['quotes']
            
        clean_data = []
        for q in quotes:
            quote_usd = q['quote']['USD']
            clean_data.append({
                'date': q['timestamp'],
                'price': quote_usd['price'],
                'volume': quote_usd['volume_24h'],
                'market_cap': quote_usd['market_cap'],
                # We can add Circulating Supply here if needed later
            })
        return clean_data
        
    except Exception as e:
        print(f"  ! Parsing error for {coin_id}: {e}")
        return []

# --- 4. NEW: Get Global Market Metrics (Macro Features) ---
def get_global_metrics(days=365):
    """
    Fetches the total market cap and BTC dominance history.
    Endpoint: /v1/global-metrics/quotes/historical
    """
    print("Fetching Global Market Metrics (Macro Data)...")
    url = f"{BASE_URL}/v1/global-metrics/quotes/historical"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'time_start': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'time_end': end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'interval': 'daily',
        'count': 10000
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()
    
    if response.status_code != 200:
        print(f"Error fetching global metrics: {data.get('status', {}).get('error_message')}")
        return []

    try:
        quotes = data['data']['quotes']
        clean_data = []
        for q in quotes:
            clean_data.append({
                'date': q['timestamp'],
                'total_market_cap': q['quote']['USD']['total_market_cap'],
                'btc_dominance': q['btc_dominance']
            })
        return clean_data
    except Exception as e:
        print(f"Error parsing global metrics: {e}")
        return []

# --- Main Execution ---
if __name__ == "__main__":
    
    print("--- Starting Full-Stack Data Collection ---")
    
    # 1. Global Metrics (Macro)
    global_data = get_global_metrics(days=365*5)
    if global_data:
        pd.DataFrame(global_data).to_csv("../../data/raw/global_metrics.csv", index=False)
        print("SUCCESS: Saved global_metrics.csv")
    else:
        print("WARNING: Could not fetch global metrics. Check API tier?")

    # 2. Top Coins List
    print("\nFetching top 50 coins...")
    top_coins = get_top_coins(limit=100)
    
    if top_coins:
        pd.DataFrame(top_coins).to_csv("../../data/raw/coin_list.csv", index=False)
        
        # 3. Metadata (Graph Edges)
        print("Fetching metadata for GNN...")
        meta_data = get_coin_metadata(top_coins)
        meta_list = []
        for cid, info in meta_data.items():
            meta_list.append({
                'id': info['id'],
                'name': info['name'],
                'symbol': info['symbol'],
                'category': info['category'],
                'tags': ",".join(info['tags']) if info['tags'] else ""
            })
        pd.DataFrame(meta_list).to_csv("../../data/raw/coin_metadata.csv", index=False)
        
        # 4. Historical Prices (Micro)
        print("Fetching historical prices...")
        all_histories = []
        
        for coin in top_coins:
            print(f"  > Fetching {coin['name']} ({coin['symbol']})...")
            history = get_historical_prices(coin['id'], days=365*5)
            
            for h in history:
                h['coin_id'] = coin['id']
                h['symbol'] = coin['symbol']
            
            all_histories.extend(history)
            time.sleep(0.5) 
            
        df_history = pd.DataFrame(all_histories)
        df_history.to_csv("../../data/raw/crypto_prices.csv", index=False)
        
        print("\nSUCCESS! All data saved to data/raw/")
        print(f"Total rows of price data: {len(df_history)}")