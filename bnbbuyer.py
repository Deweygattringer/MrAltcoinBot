#bnb buyer for trading fees
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd     # needs pip install if not installed
import numpy as np
import matplotlib.pyplot as plt   # needs pip install if not installed
import time
from helpers.parameters import (parse_args, load_config)
from helpers.handle_creds import (load_correct_creds)
from datetime import datetime
import csv
from binance.client import Client
from colorama import init
init()



# for colourful logging to the console
class txcolors:
    BUY = '\033[92m'
    WARNING = '\033[93m'
    SELL_LOSS = '\033[91m'
    SELL_PROFIT = '\033[32m'
    DIM = '\033[2m\033[35m'
    DEFAULT = '\033[39m'
    BLUE = '\033[93m'
while True:

# Load arguments then parse settings
    args = parse_args()
    DEFAULT_CREDS_FILE = 'creds.yml'
    creds_file = args.creds if args.creds else DEFAULT_CREDS_FILE
    parsed_creds = load_config(creds_file)
    access_key, secret_key = load_correct_creds(parsed_creds)
    client = Client(access_key, secret_key)
    free_bnb = client.get_asset_balance(asset='BNB')
    free_bnb_round = (float(free_bnb['free']))
    volume_bnb = float(0.03)
    date = datetime.now()
    datestring = date.strftime('%d-%m-%Y // %H:%M:%S')
   
    if free_bnb_round <= float(0.02):
        
        buy_limit = client.create_order(
            symbol = 'BNBBUSD',
            side = 'BUY',
            type = 'MARKET',
         quantity = volume_bnb
                                    )
        print(f'{txcolors.BUY}0.02 BNB successfully added{txcolors.DEFAULT}')
    else: 
        print(f'{txcolors.DEFAULT}BNB balance: {free_bnb_round} @ {datestring} \ncheking again in 3 minutes{txcolors.DEFAULT}')

    time.sleep(180)

## 