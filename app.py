import datetime

from flask import Flask, request, jsonify
import prediction

app = Flask(__name__)

@app.route('/futures/closeprice')
def hello_world():
    # Получаем значения параметров
    symbol = request.args.get('symbol')
    date = datetime.datetime.strptime(request.args.get('date'), '%Y-%m-%d')

    # Пример использования
    # symbol = 'BTCUSDT'
    interval = '1d'
    start = int(date.replace(year=date.year - 1).timestamp() * 1000)
    end = int((date - datetime.timedelta(days=1)).timestamp() * 1000)
    data = prediction.get_historical_klines(symbol, interval, start, end)
    X_train, X_test, y_train, y_test = prediction.prepare_data(data)
    model = prediction.train_model(X_train, y_train)
    next_day_price = prediction.predict_next_day_price(model, X_test)
    # return 'Hello, World!' + ' Прогноз цены на следующий день: ' + str(next_day_price)

    response_data = {
        'Success': True,
        'Date': date,
        'timestamp': date.timestamp(),
        'rate': next_day_price,
        'symbol': symbol
    }
    # # Сериализуем словарь в JSON и возвращаем как ответ
    return jsonify(response_data)


if __name__ == '__main__':
    app.run()


