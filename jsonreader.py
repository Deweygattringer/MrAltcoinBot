import json
with open('coins_bought.json') as f:
    data = json.load(f)
    bought_at = data({['BTCUSDT']:{['bought_at']}})
    print(bought_at)