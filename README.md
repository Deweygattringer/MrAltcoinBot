THE PROFIT BOT

## Description
This Binance trading bot analyses the changes in price for a given cryptoasset and trades it on every dip.
it will then continue to hold it as long as the price is rising. also, it wil hold in the event the price drops further. it will not sell unless you are in profit, at least a little bit (at least thats how it should work).



The bot will listen to changes in price for a specific asset on Binance. By default, we're only picking USDT pairs. We're excluding Margin (like BTCDOWNUSDT) and Fiat pairs.

> Information below is an example and is all configurable
- The bot checks the price every minute. if the price reaches a new higher price it will save this new high.
- The bot checks if the coin has gone down by more than x% from the previous high (ie when the price dips)
- The bot will buy with x% of your free USDT volume in this coin.
- The bot will sell only when at profit. in a bearish scenario, this means that it might take a very long time for the bot to sell.



## READ BEFORE USE
1. If you use the `TEST_MODE: False` in your config, you will be using REAL money.
2. To ensure you do not do this, ALWAYS check the `TEST_MODE` configuration item in the config.yml file..
3. This is a framework for users to modify and adapt to their overall strategy and needs, and in no way a turn-key solution.
4. You have a few options in the config file. To adapt the margins and sell criteria along with the coin you have to to this in the 'profitbot.py' file. To change the coin you have to change the tickers.txt file.
5. for the telegram bot feature(if not in use just comment it out, otherwise it will give you an error message) you need to install telegram-send as given in requirements. ( you can use "pip3 install -r requirements.txt" to install everything necessary). then you need to create a new bot with the botfather on telegram and copy the API token to telegram-send --configure . please note that you will have to reinitiate the telegram-send service when your device crashes or similar things happen.
6. Your Binance API KEY and secret go in the creds file.


## Troubleshooting

2. Open an issue on github.
    - Do not spam, do not berate, we are all humans like you, this is an open source project, not a full time job. 

## 💥 Disclaimer

All investment strategies and investments involve risk of loss. 
**Nothing contained in this program, scripts, code or repository should be construed as investment advice.**
Any reference to an investment's past or potential performance is not, 
and should not be construed as, a recommendation or as a guarantee of 
any specific outcome or profit.
By using this program you accept all liabilities, and that no claims can be made against the developers or others connected with the program.
