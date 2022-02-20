from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd     # needs pip install if not installed
import numpy as np
import matplotlib.pyplot as plt   # needs pip install if not installed
import time
from helpers.parameters import (parse_args, load_config)
from helpers.handle_creds import (load_correct_creds)
# Load arguments then parse settings
args = parse_args()
DEFAULT_CREDS_FILE = 'creds.yml'
creds_file = args.creds if args.creds else DEFAULT_CREDS_FILE
parsed_creds = load_config(creds_file)
access_key, secret_key = load_correct_creds(parsed_creds)
client = Client(access_key, secret_key)

#############################################################################################
# SIMULATOR:                                                                                #
#                                                                                           #
# line 140/142 are the SMA windows. change the strings accordingly.                         #
# valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.          #
# request historical candle (or klines)interval either every min, hr, day, week or month.   #
# starttime = '30 minutes ago UTC' for last 30 mins time.                                   #
# for live tracking: uncomment While argument and timesleep at the end and indent everything#
#############################################################################################

# while True:
symbol = 'BTCUSDT'
starttime = '20 hours ago UTC'  # to start for 1 week ago
interval = '5m'
bars = client.get_historical_klines(symbol, interval, starttime)

for line in bars:        # Keep only first 5 columns, "date" "open" "high" "low" "close"
    del line[5:]
df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
symbol_df = df

# small time Moving average. calculate 5 moving average using Pandas over close price
symbol_df['5sma'] = symbol_df['close'].rolling(5).mean()
# long time moving average. calculate 15 moving average using Pandas
symbol_df['15sma'] = symbol_df['close'].rolling(15).mean()

# To print in human readable date and time (from timestamp)
symbol_df.set_index('date', inplace=True)
symbol_df.index = pd.to_datetime(symbol_df.index, unit='ms')

# Calculate signal column 
symbol_df['Signal'] = np.where(symbol_df['5sma'] > symbol_df['15sma'], 1, 0) # ggf. hier buylogik adaptieren u.a
# Calculate position column with diff
symbol_df['Position'] = symbol_df['Signal'].diff()
    

# Add buy and sell columns
symbol_df['Buy'] = np.where(symbol_df['Position'] == 1,symbol_df['close'], np.NaN )
symbol_df['Sell'] = np.where(symbol_df['Position'] == -1,symbol_df['close'], np.NaN )


with open('outputsimulator.txt', 'w') as f:
    f.write(
            symbol_df.to_string()
               )
    
# prints current price and current sma
tickertrx = client.get_symbol_ticker(symbol=symbol)
pricetrx = tickertrx['price']    
print(f' current price: {pricetrx}')

# plot it as graphs, uncomment if not wanted

df=df.astype(float)
df[['close', '5sma','15sma']].plot()
plt.xlabel('Date',fontsize=18)
plt.ylabel(f'Close price {symbol}',fontsize=18)

plt.scatter(df.index,df['Buy'], color='green',label='Buy',  marker='^', alpha = 1) # green = buy
plt.scatter(df.index,df['Sell'], color='red',label='Sell',  marker='v', alpha = 1)  # red = sell

plt.show()

#time.sleep(60)