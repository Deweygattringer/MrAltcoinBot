import csv
# use for environment variables
import os
import math
# use if needed to pass args to external modules
import sys

import telegram_send

# used to create threads & dynamic loading of modules
import threading
import importlib

# used for directory handling
import glob

# Needed for colorful console output Install with: python3 -m pip install colorama (Mac/Linux) or pip install colorama (PC)
from colorama import init
init()

# needed for the binance API / websockets / Exception handling
from binance.client import Client
from binance.exceptions import BinanceAPIException
from requests.exceptions import ReadTimeout, ConnectionError

# used for dates
from datetime import date, datetime, timedelta
import time

# used to repeatedly execute the code
from itertools import count

# used to store trades and sell assets
import json

# Load helper modules
from helpers.parameters import (
    parse_args, load_config
)

# Load creds modules
from helpers.handle_creds import (
    load_correct_creds, test_api_key
)


# for colourful logging to the console
class txcolors:
    BUY = '\033[92m'
    WARNING = '\033[93m'
    SELL_LOSS = '\033[91m'
    SELL_PROFIT = '\033[32m'
    DIM = '\033[2m\033[35m'
    DEFAULT = '\033[39m'


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


def get_price(add_to_historical=True):
    '''Return the current price for all coins on binance'''

    global historical_prices, hsp_head

    initial_price = {}
    prices = client.get_all_tickers()

    for coin in prices:

        if CUSTOM_LIST:
            if any(item + PAIR_WITH == coin['symbol'] for item in tickers) and all(item not in coin['symbol'] for item in FIATS):
                initial_price[coin['symbol']] = { 'price': coin['price'], 'time': datetime.now()}
        else:
            if PAIR_WITH in coin['symbol'] and all(item not in coin['symbol'] for item in FIATS):
                initial_price[coin['symbol']] = { 'price': coin['price'], 'time': datetime.now()}

    if add_to_historical:
        hsp_head += 1

        if hsp_head == RECHECK_INTERVAL:
            hsp_head = 0

        historical_prices[hsp_head] = initial_price

    return initial_price


def wait_for_price():
    '''calls the initial price and ensures the correct amount of time has passed
    before reading the current price again'''

    global historical_prices, hsp_head, volatility_cooloff
   
    volatile_coins = {}
    externals = {}

    coins_up = 0
    coins_down = 0
    coins_unchanged = 0

    pause_bot()

    if historical_prices[hsp_head]['TRX' + PAIR_WITH]['time'] > datetime.now() - timedelta(minutes=float(TIME_DIFFERENCE / RECHECK_INTERVAL)):

        # sleep for exactly the amount of time required
        time.sleep((timedelta(minutes=float(TIME_DIFFERENCE / RECHECK_INTERVAL)) - (datetime.now() - historical_prices[hsp_head]['TRX' + PAIR_WITH]['time'])).total_seconds())

    print(f'Working...Session profit:{session_profit:.2f}% ')

    # retreive latest prices
    get_price()

    # calculate the difference in prices
    for coin in historical_prices[hsp_head]:

        # minimum and maximum prices over time period
        min_price = min(historical_prices, key = lambda x: float("inf") if x is None else float(x[coin]['price']))
        max_price = max(historical_prices, key = lambda x: -1 if x is None else float(x[coin]['price']))

        threshold_check = (-1.0 if min_price[coin]['time'] > max_price[coin]['time'] else 1.0) * (float(max_price[coin]['price']) - float(min_price[coin]['price'])) / float(min_price[coin]['price']) * 100

        # each coin with higher gains than our CHANGE_IN_PRICE is added to the volatile_coins dict if less than MAX_COINS is not reached.
        if threshold_check < CHANGE_IN_PRICE:
            coins_up +=1

            if coin not in volatility_cooloff:
                volatility_cooloff[coin] = datetime.now() - timedelta(minutes=TIME_DIFFERENCE)

            # only include coin as volatile if it hasn't been picked up in the last TIME_DIFFERENCE minutes already
            if datetime.now() >= volatility_cooloff[coin] + timedelta(minutes=TIME_DIFFERENCE):
                volatility_cooloff[coin] = datetime.now()

                if len(coins_bought) + len(volatile_coins) < MAX_COINS or MAX_COINS == 0:
                    volatile_coins[coin] = round(threshold_check, 3)
                    print(f'{coin} has gained - {volatile_coins[coin]}% within the last {TIME_DIFFERENCE} minutes, calculating volume in {PAIR_WITH}')

                else:
                    print(f'{txcolors.WARNING}{coin} has gained - {round(threshold_check, 3)}% within the last {TIME_DIFFERENCE} minutes, but you are holding max number of coins{txcolors.DEFAULT}')

        elif threshold_check > CHANGE_IN_PRICE:
            coins_down +=1

        else:
            coins_unchanged +=1

   
    # Here goes new code for external signalling
    externals = external_signals()
    exnumber = 0

    for excoin in externals:
        if excoin not in volatile_coins and excoin not in coins_bought and \
                (len(coins_bought) + exnumber + len(volatile_coins)) < MAX_COINS:
            volatile_coins[excoin] = 1
            exnumber +=1
            print(f'External signal received on {excoin}, calculating volume in {PAIR_WITH}')

    return volatile_coins, len(volatile_coins), historical_prices[hsp_head]


def external_signals():
    external_list = {}
    signals = {}

    # check directory and load pairs from files into external_list
    signals = glob.glob("signals/*.exs")
    for filename in signals:
        for line in open(filename):
            symbol = line.strip()
            external_list[symbol] = symbol
        try:
            os.remove(filename)
        except:
            if DEBUG: print(f'{txcolors.WARNING}Could not remove external signalling file{txcolors.DEFAULT}')

    return external_list


def pause_bot():
    '''Pause the script when exeternal indicators detect a bearish trend in the market'''
    global bot_paused, session_profit, hsp_head

    # start counting for how long the bot's been paused
    start_time = time.perf_counter()

    while os.path.isfile("signals/paused.exc"):

        if bot_paused == False:
            print(f'{txcolors.WARNING}Pausing buying due to change in market conditions, stop loss and take profit will continue to work...{txcolors.DEFAULT}')
            bot_paused = True

        # Sell function needs to work even while paused
        coins_sold = sell_coins()
        remove_from_portfolio(coins_sold)
        get_price(True)

        # pausing here
        if hsp_head == 1: print(f'Paused...Session profit:{session_profit:.2f}% Est:${(QUANTITY * session_profit)/100:.2f}')
        time.sleep((TIME_DIFFERENCE * 60) / RECHECK_INTERVAL)

    else:
        # stop counting the pause time
        stop_time = time.perf_counter()
        time_elapsed = timedelta(seconds=int(stop_time-start_time))

        # resume the bot and ser pause_bot to False
        if  bot_paused == True:
            print(f'{txcolors.WARNING}Resuming buying due to change in market conditions, total sleep time: {time_elapsed}{txcolors.DEFAULT}')
            bot_paused = False

    return


def convert_volume():
    '''Converts the volume in free USDT to the coin's volume'''

    volatile_coins, number_of_coins, last_price = wait_for_price()
    lot_size = {}
    volume = {}

    for coin in volatile_coins:

        # Find the correct step size for each coin
        # max accuracy for BTC for example is 6 decimal points
        # while XRP is only 1
        try:
            info = client.get_symbol_info(coin)
            step_size = info['filters'][2]['stepSize']
            lot_size[coin] = step_size.index('1') - 1
            
            if lot_size[coin] < 0:
                lot_size[coin] = 0
    
            
        except:
            pass
        # new code 50% vom balance free
        free_balance = client.get_asset_balance(asset='USDT')
        free = math.floor(float(free_balance['free']) * 0.9)
        # calculate the volume in coin from QUANTITY in USDT (default)
        volume[coin] = float(free / float(last_price[coin]['price']))
        
      
        # define the volume with the correct step size
        if coin not in lot_size:
            volume[coin] = float('{:.1f}'.format(volume[coin]))

        else:
            # if lot size has 0 decimal points, make the volume an integer
            if lot_size[coin] == 0:
                volume[coin] = int(volume[coin])
            else:
                volume[coin] = float('{:.{}f}'.format(volume[coin], lot_size[coin]))

    return volume, last_price

def buy():
    # check if there is enough bnb for trading fees in wallet, if not buy some
    free_bnb = client.get_asset_balance(asset='BNB')
    free_bnb_round = (float(free_bnb['free']))
    volume_bnb = float(0.03)
    print('amount bnb:', free_bnb_round)
    if free_bnb_round <= float(0.02):
        
        buy_limit = client.create_order(
            symbol = 'BNBUSDT',
            side = 'BUY',
            type = 'MARKET',
            quantity = volume_bnb
                )
        print('bnb successfully added')
    else: print('enough (more than 0.02) bnb in wallet')
    

    
    '''Place Buy market orders for each volatile coin found'''
    LastPricea = client.get_symbol_ticker(symbol='TRXUSDT')
    lastpriceb = LastPricea['price']
    volume, last_price = convert_volume()
    orders = {}
    LastPricea = client.get_symbol_ticker(symbol='TRXUSDT') 
    current = LastPricea['price']
    currentprice = float(current)
    currentprice_str = str(current)
    LastPriceb = client.get_symbol_ticker(symbol='TRXUSDT')
    currentpriceb = LastPriceb['price']
    max = str(currentpriceb)
    

    with open('current_price.txt', 'r') as file:
            btccurrent = file.readlines()[-1]
            lastcurrent = btccurrent.strip('\n').strip(' ')
            iscurrent = float(lastcurrent)
    

    with open('lastsell.txt', 'r') as file:
            lastline = file.readlines()[-1]
            lastsell = lastline.strip('\n').strip(' ')
            last_sell = float(lastsell)
    
    if current != iscurrent:
        with open('current_price.txt', 'w') as filehandle:
                for listitem in currentprice_str:
                    filehandle.write('%s' % listitem)

   
    
    with open('maxprice.txt', 'r') as file:
            btcbuy = file.readlines()[-1]
            lastb = btcbuy.strip('\n').strip(' ')
            maxpricec = float(lastb)
    
    if currentprice >= maxpricec :
    
            with open('maxprice.txt', 'w') as filehandle:
                for listitem in max:
                    filehandle.write('%s' % listitem)
    
    with open('maxprice.txt', 'r') as file:
            btcbuy = file.readlines()[-1]
            lastb = btcbuy.strip('\n').strip(' ')
            maxpricea = float(lastb)

# Hier neuer bear code
    with open('lastsell.txt', 'r') as file:
            sellline = file.readlines()[-1]
            lastsell = sellline.strip('\n').strip(' ')
            last_sell = float(lastsell)
    
    if currentprice <= last_sell :
    
            with open('lastsell.txt', 'w') as filehandle:
                for listitem in max:
                    filehandle.write('%s' % listitem)
    
    with open('lastsell.txt', 'r') as file:
            sellline = file.readlines()[-1]
            lastsell = sellline.strip('\n').strip(' ')
            last_sell = float(lastsell)

    with open('lastsellstatic.txt', 'r') as file:
            selllinestat = file.readlines()[-1]
            lastsellstat = selllinestat.strip('\n').strip(' ')
            last_sell_static = float(lastsellstat)

    with open('pricechange.txt', 'r') as file:
            changeline = file.readlines()[-1]
            changeitem = changeline.strip('\n').strip(' ')
            price_change = float(changeitem)

    historical_trx = client.get_symbol_ticker(symbol='TRXUSDT')
    histrxprice = float(historical_trx['price'])
    hstprice = str(histrxprice)
    with open('histprice.txt', 'r') as file:
            ab = file.readlines()[-1]
            ba = ab.strip('\n').strip(' ')
            hist = float(ba)
    if histrxprice != hist :
        with open('histprice.txt', 'w') as filehandle:
                for listitem in hstprice:
                    filehandle.write('%s' % listitem)
                    print('reloaded historical price')
        time.sleep(300)
    
    for coin in volume:


        # only buy if the there are no active trades on the coin
        if coin not in coins_bought and price_change <= (-0.9) and currentprice >= last_sell * 1.0070:
            print(f"{txcolors.BUY}Preparing to buy {volume[coin]} {coin}{txcolors.DEFAULT}")

            if TEST_MODE:
                orders[coin] = [{
                    'symbol': coin,
                    'orderId': 0,
                    'time': datetime.now().timestamp()
                }]

                # Log trade
                if LOG_TRADES:
                    write_log(f"Buy : {volume[coin]} {coin} - {last_price[coin]['price']}")

                continue

            # try to create a real order if the test orders did not raise an exception
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
                    boughtat_a = client.get_symbol_ticker(symbol='TRXUSDT')
                    boughtat = boughtat_a['price']
                    boughtsafe = str(boughtat)
                    rest = str('0')
                    
                    #    Log trade
                    if LOG_TRADES:
                        write_log(f"I just bought: {volume[coin]} {coin} @ {last_price[coin]['price']}")
                        
                        # reset maxprice for this buy so it will also work in more bearish trends
                        newprice = last_price[coin]['price']
                        newpricea = str(newprice)
                        with open('maxprice.txt', 'w') as filehandle:
                            for listitem in boughtsafe:
                                filehandle.write('%s' % listitem)
                        
                        #read trade log and send info to telegram bot
                        with open('trades.txt', 'r') as file:
                            logline = file.readlines()[-1]
                            lastlogbuy = logline.strip('\n').strip(' ')
                            telebuy = str(lastlogbuy)
                        telegram_send.send(messages=[telebuy])
        # hier bei petz code f+r ersten elif fall evtl auch: wenn preis seit letztem tief 0.07 prozent gestiegen ist in letzten 5min                
        elif (coin not in coins_bought and price_change >= (-0.9) and float(lastpriceb) >= last_sell_static and currentprice >= last_sell * 1.0007 and currentprice >= hist * 1.0007) or (coin not in coins_bought and price_change >= (-0.9) and last_sell_static >= currentprice and currentprice >= last_sell_static * 0.99 and currentprice >= last_sell * 1.0012) or (coin not in coins_bought and price_change >= (-0.9) and currentprice <= last_sell_static * 0.99 and currentprice >= last_sell * 1.0055) :
            print(f"{txcolors.BUY}Preparing to buy {volume[coin]} {coin}{txcolors.DEFAULT}")

            if TEST_MODE:
                orders[coin] = [{
                    'symbol': coin,
                    'orderId': 0,
                    'time': datetime.now().timestamp()
                }]

                # Log trade
                if LOG_TRADES:
                    write_log(f"Buy : {volume[coin]} {coin} - {last_price[coin]['price']}")

                continue

            # try to create a real order if the test orders did not raise an exception
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
                    boughtat_a = client.get_symbol_ticker(symbol='TRXUSDT')
                    boughtat = boughtat_a['price']
                    boughtsafe = str(boughtat)
                    rest = str('0')
                    
                    #    Log trade
                    if LOG_TRADES:
                        write_log(f"I just bought: {volume[coin]} {coin} @ {last_price[coin]['price']}")
                        
                        # reset maxprice for this buy so it will also work in more bearish trends
                        newprice = last_price[coin]['price']
                        newpricea = str(newprice)
                        with open('maxprice.txt', 'w') as filehandle:
                            for listitem in boughtsafe:
                                filehandle.write('%s' % listitem)
                        
                        #read trade log and send info to telegram bot
                        with open('trades.txt', 'r') as file:
                            logline = file.readlines()[-1]
                            lastlogbuy = logline.strip('\n').strip(' ')
                            telebuy = str(lastlogbuy)
                        telegram_send.send(messages=[telebuy])
                        
            
        else:
            print(f'Signal detected, but there is already an active trade on {coin}, or buy parameters are not met')

    return orders, last_price, volume


def sell_coins():
    '''sell coins that have reached the STOP LOSS or TAKE PROFIT threshold'''

    global hsp_head, session_profit

    last_price = get_price(False) # don't populate rolling window
    last_price = get_price(add_to_historical=True) # don't populate rolling window
    coins_sold = {}

    for coin in list(coins_bought):
       
        LastPrice = float(last_price[coin]['price'])
        BuyPrice = float(coins_bought[coin]['bought_at'])
        sell = str(LastPrice)
        PriceChange = float((LastPrice - BuyPrice) / BuyPrice * 100)
 
    
            
        with open('current_price.txt', 'w') as filehandle:
                for listitem in sell:
                    filehandle.write('%s' % listitem)
   
    
        with open('maxprice.txt', 'r') as file:
                btcbuy = file.readlines()[-1]
                lastb = btcbuy.strip('\n').strip(' ')
                maxpricea = float(lastb)
        time.sleep(5)       

        if LastPrice >= maxpricea :
                
            with open('maxprice.txt', 'w') as filehandle:
                for listitem in sell:
                    filehandle.write('%s' % listitem)
                
    

        if (LastPrice <= (maxpricea * 0.9997) and LastPrice >= (BuyPrice * 1.0018)) or (LastPrice <= BuyPrice * 0.99 ):

            print(f"{txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS}Sell criteria reached, selling {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange-(TRADING_FEE*2):.2f}% Est:${(QUANTITY*(PriceChange-(TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}")

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
                    write_log(f"I just sold: {coins_sold[coin]['volume']} {coin} @ {LastPrice} Profit: {profit:.2f} {PriceChange-(TRADING_FEE*2):.2f}%")
                    session_profit=session_profit + (PriceChange-(TRADING_FEE*2))
                    
                    #read trade log and send info to telegram bot
                    with open('trades.txt', 'r') as file:
                        loglinesell = file.readlines()[-1]
                        lastlogsell = loglinesell.strip('\n').strip(' ')
                        telesell = str(lastlogsell)
                    telegram_send.send(messages=[telesell])
                    with open('maxprice.txt', 'w') as filehandle:
                        for listitem in sell:
                            filehandle.write('%s' % listitem)
                    with open('lastsell.txt', 'w') as filehandle:
                        for listitem in sell:
                            filehandle.write('%s' % listitem)
                    with open('lastsellstatic.txt', 'w') as filehandle:
                        for listitem in sell:
                            filehandle.write('%s' % listitem)
                    profits_file = str(f"{datetime.now()}, {coins_sold[coin]['volume']}, {BuyPrice}, {LastPrice}, {profit:.2f}, {PriceChange-(TRADING_FEE*2):.2f}'\n'")
                    with open('profits.txt', 'w') as filehandle:
                        for listitem in profits_file:
                            filehandle.write('%s' % listitem)
                    PriceChangestr = str(PriceChange)
                    with open('pricechange.txt', 'w') as filehandle:
                        for listitem in PriceChangestr:
                            filehandle.write('%s' % listitem)
                    with open('histprice.txt', 'w') as filehandle:
                        for listitem in sell:
                            filehandle.write('%s' % listitem)
                    print('reloaded historical price')
            continue

        # no action; print once every TIME_DIFFERENCE
        if hsp_head == 1:
            if len(coins_bought) > 0:
                print(f'Sell criteria not yet reached, not selling {coin} for now {BuyPrice} - {LastPrice} : {txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS}{PriceChange-(TRADING_FEE*2):.2f}% Est:${(QUANTITY*(PriceChange-(TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}')

    if hsp_head == 1 and len(coins_bought) == 0: print(f'Not holding any coins')
 #neuer code
    return coins_sold

    
def update_portfolio(orders, last_price, volume):
    '''add every coin bought to our portfolio for tracking/selling later'''
    if DEBUG: print(orders)
    for coin in orders:

        coins_bought[coin] = {
            'symbol': orders[coin][0]['symbol'],
            'orderid': orders[coin][0]['orderId'],
            'timestamp': orders[coin][0]['time'],
            'bought_at': last_price[coin]['price'],
            'volume': volume[coin],
            'stop_loss': -STOP_LOSS,
            'take_profit': TAKE_PROFIT,
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
    AMERICAN_USER = parsed_config['script_options'].get('AMERICAN_USER')

    # Load trading vars
    PAIR_WITH = parsed_config['trading_options']['PAIR_WITH']
    QUANTITY = parsed_config['trading_options']['QUANTITY']
    MAX_COINS = parsed_config['trading_options']['MAX_COINS']
    FIATS = parsed_config['trading_options']['FIATS']
    TIME_DIFFERENCE = parsed_config['trading_options']['TIME_DIFFERENCE']
    RECHECK_INTERVAL = parsed_config['trading_options']['RECHECK_INTERVAL']
    CHANGE_IN_PRICE = parsed_config['trading_options']['CHANGE_IN_PRICE']
    CUSTOM_LIST = parsed_config['trading_options']['CUSTOM_LIST']
    TICKERS_LIST = parsed_config['trading_options']['TICKERS_LIST']
    TRADING_FEE = parsed_config['trading_options']['TRADING_FEE']
    SIGNALLING_MODULES = parsed_config['trading_options']['SIGNALLING_MODULES']
    STOP_LOSS = parsed_config['trading_options']['STOP_LOSS']
    TAKE_PROFIT = parsed_config['trading_options']['TAKE_PROFIT']
    if DEBUG_SETTING or args.debug:
        DEBUG = True

    # Load creds for correct environment
    access_key, secret_key = load_correct_creds(parsed_creds)

    if DEBUG:
        print(f'loaded config below\n{json.dumps(parsed_config, indent=4)}')
        print(f'Your credentials have been loaded from {creds_file}')


    # Authenticate with the client, Ensure API key is good before continuing
    if AMERICAN_USER:
        client = Client(access_key, secret_key, tld='us')
    else:
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

    # use separate files for testing and live trading
    if TEST_MODE:
        coins_bought_file_path = 'test_' + coins_bought_file_path

    # if saved coins_bought json file exists and it's not empty then load it
    if os.path.isfile(coins_bought_file_path) and os.stat(coins_bought_file_path).st_size!= 0:
        with open(coins_bought_file_path) as file:
                coins_bought = json.load(file)

    print('Press Ctrl-Q to stop the script')

    if not TEST_MODE:
        if not args.notimeout: # if notimeout skip this (fast for dev tests)
            print('WARNING: You are using the Mainnet and live funds. Waiting 1 seconds as a security measure')
            time.sleep(1)

    signals = glob.glob("signals/*.exs")
    for filename in signals:
        for line in open(filename):
            try:
                os.remove(filename)
            except:
                if DEBUG: print(f'{txcolors.WARNING}Could not remove external signalling file {filename}{txcolors.DEFAULT}')

    if os.path.isfile("signals/paused.exc"):
        try:
            os.remove("signals/paused.exc")
        except:
            if DEBUG: print(f'{txcolors.WARNING}Could not remove external signalling file {filename}{txcolors.DEFAULT}')

    # load signalling modules
    try:
        if len(SIGNALLING_MODULES) > 0:
            for module in SIGNALLING_MODULES:
                print(f'Starting {module}')
                mymodule[module] = importlib.import_module(module)
                t = threading.Thread(target=mymodule[module].do_work, args=())
                t.daemon = True
                t.start()
                time.sleep(2)
        else:
            print(f'No modules to load {SIGNALLING_MODULES}')
    except Exception as e:
        print(e)

    # seed initial prices
    get_price()
    READ_TIMEOUT_COUNT=0
    CONNECTION_ERROR_COUNT = 0
    while True:
        try:
            orders, last_price, volume = buy()
            update_portfolio(orders, last_price, volume)
            coins_sold = sell_coins()
            remove_from_portfolio(coins_sold)
        except ReadTimeout as rt:
            READ_TIMEOUT_COUNT += 1
            print(f'{txcolors.WARNING}We got a timeout error from from binance. Going to re-loop. Current Count: {READ_TIMEOUT_COUNT}\n{rt}{txcolors.DEFAULT}')
        except ConnectionError as ce:
            CONNECTION_ERROR_COUNT +=1 
            print(f'{txcolors.WARNING}We got a timeout error from from binance. Going to re-loop. Current Count: {CONNECTION_ERROR_COUNT}\n{ce}{txcolors.DEFAULT}')

