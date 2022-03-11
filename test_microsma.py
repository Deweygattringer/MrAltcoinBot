#test

# lowbot_Micro Sma v2.6.py

from distutils.command.build_py import build_py
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import pandas as pd     # needs pip install if not installed
import numpy as np
import matplotlib.pyplot as plt   # needs pip install if not installed
import os
import math
import sys
import telegram_send
from colorama import init
init()
from binance.client import Client
from binance.exceptions import BinanceAPIException
from requests.exceptions import ReadTimeout, ConnectionError
from datetime import date, datetime, timedelta
import time
from itertools import count
import json
from helpers.parameters import (
    parse_args, load_config
)
from helpers.handle_creds import (
    load_correct_creds, test_api_key
)

class txcolors:
    BUY = '\033[92m'
    WARNING = '\033[93m'
    SELL_LOSS = '\033[91m'
    SELL_PROFIT = '\033[32m'
    DIM = '\033[2m\033[35m'
    DEFAULT = '\033[39m'
    BLUE = '\033[93m'
# tracks profit/loss each session
global session_profit
session_profit = 0

# print with timestamps
old_out = sys.stdout
class St_ampe_dOut:
    """Stamped stdout."""
    nl = True
    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            old_out.write(x)
            self.nl = True
        elif self.nl:
            old_out.write(f'{txcolors.DIM}[{str(datetime.now().replace(microsecond=0))}]{txcolors.DEFAULT} {x}')
            self.nl = False
        else:
            old_out.write(x)

    def flush(self):
        pass

sys.stdout = St_ampe_dOut()




    
def buy():

	
    orders = {}
    lot_size = {}
    volume = {}
    last_price = {}
    CurrentSymbolPrice = client.get_symbol_ticker(symbol= 'TRXBUSD')
    Price = CurrentSymbolPrice['price']
    currentprice = float(Price)
    currentstring= str(Price)
    
    #CONVERT VOLUME OF COIN TO BUY
    for coin in tickers:
        last_price[coin] = CurrentSymbolPrice

        # Find the correct step size for each coin
        # max accuracy for TRX for example is 6 decimal points, while XRP just 1.
        try:
            info = client.get_symbol_info(coin)
            step_size = info['filters'][2]['stepSize']
            lot_size[coin] = step_size.index('1') - 1
            
            if lot_size[coin] < 0:
                lot_size[coin] = 0
        
        except:
            pass
        # new code x% of free balance
        free_balance = client.get_asset_balance(asset='BUSD')
        free = math.floor(float(free_balance['free']) * 0.5)
        # calculate the volume in coin from QUANTITY in 'PAIRWITH' (default)
        volume[coin] = float(free / float(Price))
        
        # define the volume with the correct step size
        if coin not in lot_size:
            volume[coin] = float('{:.1f}'.format(volume[coin]))

        else:
            # if lot size has 0 decimal points, make the volume an integer
            if lot_size[coin] == 0:
                volume[coin] = int(volume[coin])
            else:
                volume[coin] = float('{:.{}f}'.format(volume[coin], lot_size[coin]))





    # GET HISTORICAL PRICE DATA
    # # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    avg_price = client.get_avg_price(symbol='TRXBUSD')
    avg = float(avg_price['price'])
    
    symbol = 'TRXBUSD'
    starttime = '24 hours ago UTC'  # to start for 1 week ago
    interval = '15m'
    bars = client.get_historical_klines(symbol, interval, starttime)
    for line in bars:        # Keep only first 5 columns, "date" "open" "high" "low" "close"
        del line[5:]
    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
    symbol_df = df
# ultra small time Moving average. 
    symbol_df['ussma'] = symbol_df['close'].rolling(1).mean()
    ussma = symbol_df.loc[(symbol_df.shape[0]-1), 'ussma']
    oldussma = symbol_df.loc[(symbol_df.shape[0]-2), 'ussma']
    veryoldussma = symbol_df.loc[(symbol_df.shape[0]-3), 'ussma']
# small time Moving average. 
    symbol_df['ssma'] = symbol_df['close'].rolling(3).mean()
    ssma = symbol_df.loc[(symbol_df.shape[0]-1), 'ssma']
    oldssma = symbol_df.loc[(symbol_df.shape[0]-2), 'ssma']
    veryoldssma = symbol_df.loc[(symbol_df.shape[0]-3), 'ssma']
#medium time Moving average.
    symbol_df['msma'] = symbol_df['close'].rolling(6).mean()
    msma = symbol_df.loc[(symbol_df.shape[0]-1), 'msma']
    oldmsma = symbol_df.loc[(symbol_df.shape[0]-2), 'msma']
    veryoldmsma = symbol_df.loc[(symbol_df.shape[0]-3), 'msma']
# print in human readable date and time (from timestamp)
    symbol_df.set_index('date', inplace=True)
    symbol_df.index = pd.to_datetime(symbol_df.index, unit='ms')
# Calculate signal column 
    symbol_df['Signal'] = np.where(symbol_df['ssma'] > symbol_df['msma'], 1, 0) 
    signal = symbol_df['Signal'].array[-1]
    signalold = symbol_df['Signal'].array[-2]
# Calculate position column with diff
    symbol_df['Position'] = symbol_df['Signal'].diff()
    position = symbol_df['Position'].array[-1]
    positionold = symbol_df['Position'].array[-2]

# Add buy- and sellprice columns
    symbol_df['Buy'] = np.where(symbol_df['Position'] == 1,symbol_df['close'], np.NaN )
    symbol_df['Sell'] = np.where(symbol_df['Position'] == -1,symbol_df['close'], np.NaN )
    #print(f' s {signal} so{signalold} p {position}po {positionold}')
    with open('output.txt', 'w') as f:
        f.write(
                symbol_df.to_string()
               )
    


    # pricechanges from very old to old
    #((new-old)/old)*100
    changeoldssma = (((oldssma - veryoldssma)/ veryoldssma)*100)
    changeoldmsma = (((oldmsma - veryoldmsma)/ veryoldmsma)*100)
    changeoldussma = (((oldussma - veryoldussma)/ veryoldussma)*100)
    #price changes from old to current sma 
    changessma = (((ssma - oldssma)/oldssma)*100)
    changemsma = (((msma - oldmsma)/oldmsma)*100)
    changeussma = (((ussma - oldussma)/oldussma)*100)
    
   
    
    
    with open('maxprice.txt', 'r') as file:
            btcbuy = file.readlines()[-1]
            lastb = btcbuy.strip('\n').strip(' ')
            maxprice = float(lastb)
            maxpricestring = str(maxprice)
    if currentprice >= maxprice :
    
            with open('maxprice.txt', 'w') as filehandle:
                for listitem in currentstring:
                    filehandle.write('%s' % listitem)
    
    
    print(f'{txcolors.BLUE}SIG: {signal}/ SIG_OLD : {signalold} / POS: {position} /  / POS_OLD: {positionold}{txcolors.DEFAULT}\n ')
    print(f'{txcolors.BLUE}ussma : {ussma}      ssma : {ssma}          msma : {msma}   {txcolors.DEFAULT} \n ')  
    print(f'{txcolors.BLUE}current : {currentprice}      avg : {avg}      maxprice: {maxprice}    {txcolors.DEFAULT} \n  ') 
    print(f'{txcolors.BLUE}ussmachange : {changeussma}   smachng : {changessma}       msmachng : {changemsma}   {txcolors.DEFAULT} \n       ')  
    print(f'Coin to buy: {coin}\n')
    print(f'volume: {volume[coin]}\n')
    print(last_price) 
    
    
    for coin in tickers:
        
        buysignal = (coin not in coins_bought and position == 1.0 and signal == 1 and avg > ssma and ussma > ssma  and ussma > oldussma * 1.0003 and ssma > oldssma)

        buysignal2 = (coin not in coins_bought and positionold == 0.0 and signalold == 1 and signal == 1  and avg > ssma and ussma > ssma  and ussma > oldussma * 1.0003 and ssma > oldssma )
        print(f'Buysignal = position is 1.0 and signal is 1 and avg > ssma and ussma > ssma and curr > avg and ussma > oldussma and ssma > oldssma:\n{buysignal}\n ')
        print(f'Buysignal2 = positionold is 0.0 and signalold is 1 and signal is 1 and avg > ssma and ussma > ssma and curr > avg and ussma > oldussma and ssma > oldssma: \n{buysignal2}\n ')
    
        # only buy if the there are no active trades on the coin
        if buysignal:
            print(f"{txcolors.BUY}Preparing to buy {[coin]} {coin}{txcolors.DEFAULT}")

            try:
                buy_limit = client.create_order(
                    symbol = coin,
                    side = 'BUY',
                    type = 'MARKET',
                    quantity = volume[coin]
                )

            # error handling here in case position cannot be placed
            except Exception as e:
                print(e)

            # run the else block if the position has been placed and return order info
            else:
                orders[coin] = client.get_all_orders(symbol=coin, limit=1)

                # binance sometimes returns an empty list, the code will wait here until binance returns the order
                while orders[coin] == []:
                    print('Binance is being slow in returning the order, calling the API again...')

                    orders[coin] = client.get_all_orders(symbol=coin, limit=1)
                    time.sleep(1)

                else:
                    print('Order returned, saving order to file')
                    
                    #    Log trade
                    if LOG_TRADES:
                        write_log(f"lowbot just bought position=1: {volume[coin]} {coin} @ {currentprice}")
                        
                        #read tradelog and send buy info on telegram
                        with open('trades.txt', 'r') as file:
                            logline = file.readlines()[-1]
                            lastlogbuy = logline.strip('\n').strip(' ')
                            telebuy = str(lastlogbuy)
                            telegram_send.send(messages=[telebuy])
                        with open('maxprice.txt', 'w') as filehandle:
                            for listitem in currentstring:
                                filehandle.write('%s' % listitem)
                        
        elif buysignal2:
            print(f"{txcolors.BUY}Preparing to buy {volume[coin]} {coin}{txcolors.DEFAULT}")

            try:
                buy_limit = client.create_order(
                    symbol = coin,
                    side = 'BUY',
                    type = 'MARKET',
                    quantity = volume[coin]
                )

            # error handling here in case position cannot be placed
            except Exception as e:
                print(e)

            # run the else block if the position has been placed and return order info
            else:
                orders[coin] = client.get_all_orders(symbol=coin, limit=1)

                # binance sometimes returns an empty list, the code will wait here until binance returns the order
                while orders[coin] == []:
                    print('Binance is being slow in returning the order, calling the API again...')

                    orders[coin] = client.get_all_orders(symbol=coin, limit=1)
                    time.sleep(1)

                else:
                    print('Order returned, saving order to file')
                    
                    #    Log trade
                    if LOG_TRADES:
                        write_log(f"lowbot just bought position 0, signal 1: {volume[coin]} {coin} @ {currentprice}")
                        
                        #read tradelog and send buy info on telegram
                        with open('trades.txt', 'r') as file:
                            logline = file.readlines()[-1]
                            lastlogbuy = logline.strip('\n').strip(' ')
                            telebuy = str(lastlogbuy)
                            telegram_send.send(messages=[telebuy])
                        with open('maxprice.txt', 'w') as filehandle:
                            for listitem in currentstring:
                                filehandle.write('%s' % listitem)

        elif   (coin  in coins_bought and position == 1.0 and signal == 1.0 and avg > ussma and ussma > ssma and currentprice > avg and ussma > oldussma and ssma > oldssma):

            print(f'Buy parameters met, but there is already an active trade')
        elif   (coin in coins_bought and position == 1.0 and signal == 1.0 and avg > ussma and ussma > ssma and currentprice > avg and ussma > oldussma and ssma > oldssma):

            print(f'Buy parameters met, but there is already an active trade')
        else:
            print(f'Buy parameters not yet met')

    return orders,  volume, last_price

def sell_coins():
    '''sell coins that have reached the STOP LOSS or TAKE PROFIT threshold'''

    global hsp_head, session_profit

    coins_sold = {}
    
 


    for coin in list(coins_bought):
        CurrentSymbolPrice = client.get_symbol_ticker(symbol= 'TRXBUSD')
        Price = CurrentSymbolPrice['price']
        LastPrice = float(Price)
        BuyPrice = float(coins_bought[coin]['bought_at'])
        PriceChange = float((LastPrice - BuyPrice) / BuyPrice * 100)

        currentstring= str(LastPrice)

    # track the sma 'interval' * X ago
        with open('output.txt', 'r') as file:
            outputfile = file.readlines()
            oldsmaline = outputfile[-2].split()
            oldussma= float(oldsmaline[-7])
            oldssma= float(oldsmaline[-6])
            oldmsma= float(oldsmaline[-5])
        with open('output.txt', 'r') as file:   
            veryoldsmaline = outputfile[-3].split()
            veryoldussma= float(veryoldsmaline[-7])
            veryoldssma= float(veryoldsmaline[-6])
            veryoldmsma= float(veryoldsmaline[-5])
        with open('output.txt', 'r') as file:
        # track the current sma
            outputfile = file.readlines()
            outputline = outputfile[-1].split()
            ussma =(float(outputline[-7]))
            ssma =(float(outputline[-6]))
            msma = (float(outputline[-5]))
            signal = float(outputline[-4])
            position = float(outputline[-3])
            msmastring = str(msma)   
            ssmastring = str(ssma)
    
        avg_price = client.get_avg_price(symbol='TRXBUSD')
        avg = float(avg_price['price'])
    # pricechanges from very old to old
    #((new-old)/old)*100
        changeoldssma = (((oldssma - veryoldssma)/ veryoldssma)*100)
        changeoldmsma = (((oldmsma - veryoldmsma)/ veryoldmsma)*100)

    #price changes from old to current sma 
        changessma = (((ssma - oldssma)/oldssma)*100)
        changemsma = (((msma - oldmsma)/oldmsma)*100)
        with open('maxprice.txt', 'r') as file:
            btcbuy = file.readlines()[-1]
            lastb = btcbuy.strip('\n').strip(' ')
            maxprice = float(lastb)
            maxpricestring = str(maxprice)
        profit = ((LastPrice - BuyPrice) * coins_bought[coin]['volume'])* (1-(TRADING_FEE*2)) # adjust for trading fee here


        # sell logic
        stoploss = LastPrice < BuyPrice * 1.001 and avg < ussma and ussma < oldussma and ssma < oldssma and ssma < msma
        sell = ((LastPrice <= maxprice * 0.9996) and (LastPrice >= BuyPrice * 1.0017))
        
     
        print(f'sell = LastPrice {LastPrice} <= maxprice  * 0.9996 {maxprice * 0.9996}) and (LastPrice {LastPrice} >= BuyPrice * 1.0017 {BuyPrice * 1.0017}): {sell} \n')
        print(f'stoploss = curr{LastPrice} < Buy+ 0.1{BuyPrice * 1.001} and avg {avg} < ussma {ussma}  and ussma{ussma} <  oldussma{oldussma} and ssma {ssma} < oldssma{oldssma} and ssma{ssma} < msma{msma}: {stoploss}\n ')
        print(f'profit at the moment = {profit} %\n')

        if sell:  

            print(f"{txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS}Sell criteria reached, selling {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange-(TRADING_FEE*2):.2f}% Est:${((PriceChange-(TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}")

            # try to create a real order
            try:

                if not TEST_MODE:
                    sell_coins_limit = client.create_order(
                        symbol = coin,
                        side = 'SELL',
                        type = 'MARKET',
                        quantity = coins_bought[coin]['volume']

                    )

            # error handling here in case position cannot be placed
            except Exception as e:
                print(e)

            # run the else block if coin has been sold and create a dict for each coin sold
            else:
                coins_sold[coin] = coins_bought[coin]

                # prevent system from buying this coin for the next TIME_DIFFERENCE minutes
                volatility_cooloff[coin] = datetime.now()

                # Log trade
                if LOG_TRADES:
                    profit = ((LastPrice - BuyPrice) * coins_sold[coin]['volume'])* (1-(TRADING_FEE*2)) # adjust for trading fee here
                    write_log(f"lowbot just sold with profit_sell_opt: {coins_sold[coin]['volume']} {coin} @ {LastPrice} Profit: {profit:.2f} {PriceChange-(TRADING_FEE*2):.2f}% --- sell = curr {LastPrice}< max{maxprice *0.9996} and curr {LastPrice} >= buy+prof {BuyPrice * 1.0017}")
                    session_profit=session_profit + (PriceChange-(TRADING_FEE*2))
                    
                    profits_file = str(f" {datetime.now()}, {coins_sold[coin]['volume']}, {BuyPrice}, {LastPrice}, {profit:.2f}, {PriceChange-(TRADING_FEE*2):.2f} \n")
                    # create a proftis/tradelog ready to be importet to excel
                    with open('profits.txt', 'a') as filehandle:
                        for listitem in profits_file:
                            filehandle.write('%s' % listitem)
                    # read logline and send sell info on telegram
                    with open('trades.txt', 'r') as file:
                            logline = file.readlines()[-1]
                            lastlogbuy = logline.strip('\n').strip(' ')
                            telebuy = str(lastlogbuy)
                            telegram_send.send(messages=[telebuy])
                    with open('maxprice.txt', 'w') as filehandle:
                        for listitem in currentstring:
                            filehandle.write('%s' % listitem)
                    
                    
            continue

       
        elif stoploss:  

            print(f"{txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS}Sell criteria reached, selling {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange-(TRADING_FEE*2):.2f}% Est:${((PriceChange-(TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}")

            # try to create a real order
            try:

                if not TEST_MODE:
                    sell_coins_limit = client.create_order(
                        symbol = coin,
                        side = 'SELL',
                        type = 'MARKET',
                        quantity = coins_bought[coin]['volume']

                    )

            # error handling here in case position cannot be placed
            except Exception as e:
                print(e)

            # run the else block if coin has been sold and create a dict for each coin sold
            else:
                coins_sold[coin] = coins_bought[coin]

                # prevent system from buying this coin for the next TIME_DIFFERENCE minutes
                volatility_cooloff[coin] = datetime.now()

                # Log trade
                if LOG_TRADES:
                    profit = ((LastPrice - BuyPrice) * coins_sold[coin]['volume'])* (1-(TRADING_FEE*2)) # adjust for trading fee here
                    write_log(f"lowbot just sold with oussma < ossma({oldussma} < {oldssma}):  {coins_sold[coin]['volume']} {coin} @ {LastPrice} Profit: {profit:.2f} {PriceChange-(TRADING_FEE*2):.2f}%")
                    session_profit=session_profit + (PriceChange-(TRADING_FEE*2))
                    
                    
                    profits_file = str(f" {datetime.now()}, {coins_sold[coin]['volume']}, {BuyPrice}, {LastPrice}, {profit:.2f}, {PriceChange-(TRADING_FEE*2):.2f} \n")
                    # create a proftis/tradelog ready to be importet to excel
                    with open('profits.txt', 'a') as filehandle:
                        for listitem in profits_file:
                            filehandle.write('%s' % listitem)
                    # read logline and send sell info on telegram
                    with open('trades.txt', 'r') as file:
                            logline = file.readlines()[-1]
                            lastlogbuy = logline.strip('\n').strip(' ')
                            telebuy = str(lastlogbuy)
                            telegram_send.send(messages=[telebuy])
                    with open('maxprice.txt', 'w') as filehandle:
                        for listitem in currentstring:
                            filehandle.write('%s' % listitem)
            continue
       
    
        # no action; print once every TIME_DIFFERENCE
        if hsp_head == 1:
            if len(coins_bought) > 0:
                print(f'Sell criteria not yet reached, not selling {coin} for now {BuyPrice} - {LastPrice} : {txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS}{PriceChange-(TRADING_FEE*2):.2f}% Est:${((PriceChange-(TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}')

    if hsp_head == 1 and len(coins_bought) == 0: print(f'No active trade at the moment')

    return coins_sold

def update_portfolio(orders, last_price):
    '''add every coin bought to our portfolio for tracking/selling later'''
    if DEBUG: print(orders)
    for coin in orders:

        coins_bought[coin] = {
            'symbol': orders[coin][0]['symbol'],
            'orderid': orders[coin][0]['orderId'],
            'timestamp': orders[coin][0]['time'],
            'bought_at': last_price[coin]['price'],
            'volume': volume[coin],
            }

        # save the coins in a json file in the same directory
        with open(coins_bought_file_path, 'w') as file:
            json.dump(coins_bought, file, indent=4)
                           
        print(f'Order with id {orders[coin][0]["orderId"]} placed and saved to file')

def remove_from_portfolio(coins_sold):
    '''Remove coins sold due to SL or TP from portfolio'''
    for coin in coins_sold:
        coins_bought.pop(coin)

    with open(coins_bought_file_path, 'w') as file:
        json.dump(coins_bought, file, indent=4)

def write_log(logline):
    timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
    with open(LOG_FILE,'a+') as f:
        f.write(timestamp + ' ' + logline + '\n')

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    # Load arguments then parse settings
    args = parse_args()
    mymodule = {}
    # set to false at Start
    global bot_paused
    bot_paused = False
    DEFAULT_CONFIG_FILE = 'config.yml'
    DEFAULT_CREDS_FILE = 'creds.yml'

    config_file = args.config if args.config else DEFAULT_CONFIG_FILE
    creds_file = args.creds if args.creds else DEFAULT_CREDS_FILE
    parsed_config = load_config(config_file)
    parsed_creds = load_config(creds_file)

    # Default no debugging
    DEBUG = False

    # Load system vars
    TEST_MODE = parsed_config['script_options']['TEST_MODE']
    LOG_TRADES = parsed_config['script_options'].get('LOG_TRADES')
    LOG_FILE = parsed_config['script_options'].get('LOG_FILE')
    DEBUG_SETTING = parsed_config['script_options'].get('DEBUG')
  
    # Load trading vars
    PAIR_WITH = parsed_config['trading_options']['PAIR_WITH']
    MAX_COINS = parsed_config['trading_options']['MAX_COINS']
    FIATS = parsed_config['trading_options']['FIATS']
    TIME_DIFFERENCE = parsed_config['trading_options']['TIME_DIFFERENCE']
    RECHECK_INTERVAL = parsed_config['trading_options']['RECHECK_INTERVAL']
    CHANGE_IN_PRICE = parsed_config['trading_options']['CHANGE_IN_PRICE']
    CUSTOM_LIST = parsed_config['trading_options']['CUSTOM_LIST']
    TICKERS_LIST = parsed_config['trading_options']['TICKERS_LIST']
    TRADING_FEE = parsed_config['trading_options']['TRADING_FEE']
    
    if DEBUG_SETTING or args.debug:
        DEBUG = True

    # Load creds for correct environment
    access_key, secret_key = load_correct_creds(parsed_creds)
    
    if DEBUG:
        print(f'loaded config below\n{json.dumps(parsed_config, indent=4)}')
        print(f'Your credentials have been loaded from {creds_file}')
    client = Client(access_key, secret_key)
        
    # If the users has a bad / incorrect API key.
    api_ready, msg = test_api_key(client, BinanceAPIException)
    if api_ready is not True:
       exit(f'{txcolors.SELL_LOSS}{msg}{txcolors.DEFAULT}')

    # Use CUSTOM_LIST symbols if CUSTOM_LIST is set to True
    if CUSTOM_LIST: tickers=[line.strip() for line in open(TICKERS_LIST)]

    # try to load all the coins bought by the bot if the file exists and is not empty
    coins_bought = {}

    # path to the saved coins_bought file
    coins_bought_file_path = 'coins_bought.json'
    
    # rolling window of prices; cyclical queue
    historical_prices = [None] * (TIME_DIFFERENCE * RECHECK_INTERVAL)
    hsp_head = -1

    # prevent including a coin in volatile_coins if it has already appeared there less than TIME_DIFFERENCE minutes ago
    volatility_cooloff = {}
    
    # if saved coins_bought json file exists and it's not empty then load it
    if os.path.isfile(coins_bought_file_path) and os.stat(coins_bought_file_path).st_size!= 0:
        with open(coins_bought_file_path) as file:
                coins_bought = json.load(file)


    if not TEST_MODE:
        if not args.notimeout: # if notimeout skip this (fast for dev tests)
            print('WARNING: You are using the Mainnet and live funds.')
            
    
    # seed initial prices
  #  get_price()
    READ_TIMEOUT_COUNT=0
    CONNECTION_ERROR_COUNT = 0
    while True:
        try:
            
            orders, volume, last_price = buy()
            update_portfolio(orders, volume,)
            
            coins_sold = sell_coins()
            remove_from_portfolio(coins_sold)
            print(f'Working...Session profit:{session_profit:.2f}% ')
            time.sleep(2)
            
            
        except ReadTimeout as rt:
            READ_TIMEOUT_COUNT += 1
            print(f'{txcolors.WARNING}We got a timeout error from from binance. Going to re-loop. Current Count: {READ_TIMEOUT_COUNT}\n{rt}{txcolors.DEFAULT}')
        except ConnectionError as ce:
            CONNECTION_ERROR_COUNT +=1 
            print(f'{txcolors.WARNING}We got a timeout error from from binance. Going to re-loop. Current Count: {CONNECTION_ERROR_COUNT}\n{ce}{txcolors.DEFAULT}')
# test micro sma v2.6 // 10.3.2022 
