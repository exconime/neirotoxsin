from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go
import tempfile
import os

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def plot_graph(data, y_column, predictions):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data[y_column], y=predictions, mode='markers',
                             marker=dict(color='#324f59'), name='Точки данных'))

    fig.add_trace(go.Scatter(x=data[y_column], y=predictions, mode='lines',
                             line=dict(color='#243940'), name='Линия тренда'))

    fig.update_layout(
        title='Предсказания модели в сравнении с фактическими значениями',
        xaxis_title='Фактические значения',
        yaxis_title='Прогнозируемые значения',
        plot_bgcolor='#152125',
        paper_bgcolor='#152125',
        font=dict(color='white', family='Montserrat')
    )

    return fig.to_html()


def process_file(file, y_column):
    try:
        data = pd.read_csv(file)
        data.ffill(inplace=True)

        Y = data[y_column]

        # Выбираем только числовые столбцы
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Удаляем столбец Y из списка независимых переменных
        if y_column in numeric_columns:
            numeric_columns.remove(y_column)

        X = data[numeric_columns].values.astype(np.float32)
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()

        predictions = model.predict(X)
        data['predictions'] = predictions

        print("\nСводка модели:")
        print(model.summary())

        epsilon = 1e-10
        absolute_percentage_errors = np.abs((Y - predictions) / (Y + epsilon))
        mape = np.mean(absolute_percentage_errors) * 100
        mse = mean_squared_error(Y, predictions)
        mae = mean_absolute_error(Y, predictions)
        r2 = r2_score(Y, predictions)

        graph_html = plot_graph(data, y_column, predictions)

        return mape, mse, mae, r2, graph_html

    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        return None, None, None, None, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')

    if file is None or file.filename == '':
        return "Вы не выбрали файл."

    y_column = request.form.get('y_column')
    if not y_column:
        return "Вы не указали название столбца по которому будет произведен прогноз."

    mape, mse, mae, r2, graph_html = process_file(file, y_column)

    if mse is not None:
        return render_template('result.html', mape=mape, mse=mse, mae=mae, r2=r2, graph_html=graph_html)
    else:
        return "Во время обработки данных произошла ошибка."


if __name__ == '__main__':
    app.run(debug=True)
