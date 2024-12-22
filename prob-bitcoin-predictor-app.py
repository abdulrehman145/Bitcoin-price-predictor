import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QTabWidget)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class BitcoinPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitcoin Price Predictor")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        
    def initUI(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create header
        header = QLabel("Bitcoin Price Predictor")
        header.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
        """)
        layout.addWidget(header)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Add tabs
        self.add_data_visualization_tab(tab_widget)
        self.add_prediction_tab(tab_widget)
        self.add_model_comparison_tab(tab_widget)
        self.add_model_evaluation_tab(tab_widget)  # Add the evaluation tab
        
    def add_model_evaluation_tab(self, tab_widget):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Evaluation Button
        evaluate_btn = QPushButton("Evaluate All Models")
        evaluate_btn.clicked.connect(self.evaluate_models)
        evaluate_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        layout.addWidget(evaluate_btn)
        
        # Results area
        self.evaluation_results_label = QLabel("Evaluation results will appear here")
        self.evaluation_results_label.setStyleSheet("""
            QLabel { 
                font-size: 16px; 
                padding: 20px; 
                background-color: #f5f6fa; 
                border-radius: 4px; 
            }
        """)
        layout.addWidget(self.evaluation_results_label)

        tab_widget.addTab(tab, "Model Evaluation")

    def add_data_visualization_tab(self, tab_widget):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create plot buttons
        buttons_layout = QHBoxLayout()
        plot_types = ["Price History", "Candlestick", "Volume Distribution", 
                     "Price vs Volume", "Volume by Month"]
        
        for plot_type in plot_types:
            btn = QPushButton(plot_type)
            btn.clicked.connect(lambda checked, t=plot_type: self.update_plot(t))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            buttons_layout.addWidget(btn)
            
        layout.addLayout(buttons_layout)
        
        # Create matplotlib figure
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        tab_widget.addTab(tab, "Data Visualization")

    def add_prediction_tab(self, tab_widget):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Linear Regression", "ARIMA", "SARIMA", 
                                   "LSTM", "Gradient Boosting"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        
        # Predict button
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.make_prediction)
        predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        model_layout.addWidget(predict_btn)
        
        layout.addLayout(model_layout)
        
        # Results area
        self.results_label = QLabel("Results will appear here")
        self.results_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 20px;
                background-color: #f5f6fa;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.results_label)
        
        # Prediction plot
        self.pred_figure = plt.figure(figsize=(10, 6))
        self.pred_canvas = FigureCanvas(self.pred_figure)
        layout.addWidget(self.pred_canvas)
        
        tab_widget.addTab(tab, "Price Prediction")

    def evaluate_models(self):
        try:
            # Load the data
            data = self.load_data()
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]

            # Store evaluation metrics for all models
            evaluation_results = ""

            # Linear Regression
            model_lr = LinearRegression()
            X_train_lr = np.array(train_data.index.map(lambda x: x.toordinal())).reshape(-1, 1)
            y_train_lr = train_data['Close'].values
            model_lr.fit(X_train_lr, y_train_lr)
            X_test_lr = np.array(test_data.index.map(lambda x: x.toordinal())).reshape(-1, 1)
            predictions_lr = model_lr.predict(X_test_lr)
            mse_lr = mean_squared_error(test_data['Close'], predictions_lr)
            r2_lr = r2_score(test_data['Close'], predictions_lr)
            evaluation_results += f"Linear Regression: MSE = {mse_lr:.2f}, R2 = {r2_lr:.2f}\n"

            # Gradient Boosting
            model_gbr = GradientBoostingRegressor()
            model_gbr.fit(X_train_lr, y_train_lr)
            predictions_gbr = model_gbr.predict(X_test_lr)
            mse_gbr = mean_squared_error(test_data['Close'], predictions_gbr)
            r2_gbr = r2_score(test_data['Close'], predictions_gbr)
            evaluation_results += f"Gradient Boosting: MSE = {mse_gbr:.2f}, R2 = {r2_gbr:.2f}\n"

            # ARIMA
            model_arima = ARIMA(train_data['Close'], order=(5, 1, 0))
            model_arima_fit = model_arima.fit()
            predictions_arima = model_arima_fit.forecast(steps=len(test_data))
            mse_arima = mean_squared_error(test_data['Close'], predictions_arima)
            r2_arima = r2_score(test_data['Close'], predictions_arima)
            evaluation_results += f"ARIMA: MSE = {mse_arima:.2f}, R2 = {r2_arima:.2f}\n"

            # SARIMA
            model_sarima = SARIMAX(train_data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            model_sarima_fit = model_sarima.fit(disp=False)
            predictions_sarima = model_sarima_fit.forecast(steps=len(test_data))
            mse_sarima = mean_squared_error(test_data['Close'], predictions_sarima)
            r2_sarima = r2_score(test_data['Close'], predictions_sarima)
            evaluation_results += f"SARIMA: MSE = {mse_sarima:.2f}, R2 = {r2_sarima:.2f}\n"

            # LSTM
            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train_data[['Close']])
            scaled_test = scaler.transform(test_data[['Close']])
            X_train_lstm, y_train_lstm = [], []
            for i in range(60, len(scaled_train)):
                X_train_lstm.append(scaled_train[i-60:i, 0])
                y_train_lstm.append(scaled_train[i, 0])
            X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)
            X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))

            model_lstm = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            model_lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=0)

            X_test_lstm, y_test_lstm = [], []
            for i in range(60, len(scaled_test)):
                X_test_lstm.append(scaled_test[i-60:i, 0])
            X_test_lstm = np.array(X_test_lstm)
            X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
            predictions_lstm = model_lstm.predict(X_test_lstm)
            predictions_lstm = scaler.inverse_transform(predictions_lstm)
            mse_lstm = mean_squared_error(test_data['Close'][60:], predictions_lstm[:, 0])
            r2_lstm = r2_score(test_data['Close'][60:], predictions_lstm[:, 0])
            evaluation_results += f"LSTM: MSE = {mse_lstm:.2f}, R2 = {r2_lstm:.2f}\n"

            # Display evaluation results
            self.evaluation_results_label.setText(f"Model evaluation completed.\n{evaluation_results}")

        except Exception as e:
            self.evaluation_results_label.setText(f"Evaluation error: {str(e)}")
               
    def add_model_comparison_tab(self, tab_widget):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Compare button
        compare_btn = QPushButton("Compare All Models")
        compare_btn.clicked.connect(self.compare_models)
        compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        layout.addWidget(compare_btn)
        
        # Comparison plot
        self.comp_figure = plt.figure(figsize=(10, 6))
        self.comp_canvas = FigureCanvas(self.comp_figure)
        layout.addWidget(self.comp_canvas)
        
        tab_widget.addTab(tab, "Model Comparison")
        
    def load_data(self):
        # Load your Bitcoin data here
        # For now, using sample data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'Close': np.random.normal(40000, 5000, len(dates)),
            'Open': np.random.normal(40000, 5000, len(dates)),
            'High': np.random.normal(41000, 5000, len(dates)),
            'Low': np.random.normal(39000, 5000, len(dates)),
            'Volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        return data
        
    def update_plot(self, plot_type):
        self.figure.clear()
        data = self.load_data()
        ax = self.figure.add_subplot(111)
        
        if plot_type == "Price History":
            ax.plot(data.index, data['Close'])
            ax.set_title('Bitcoin Price History')
        elif plot_type == "Candlestick":
            # Implement candlestick plot
            ax.plot(data.index, data[['Open', 'Close', 'High', 'Low']])
            ax.set_title('Bitcoin Candlestick Chart')
        elif plot_type == "Volume Distribution":
            ax.hist(data['Volume'], bins=50)
            ax.set_title('Volume Distribution')
        elif plot_type == "Price vs Volume":
            ax.scatter(data['Close'], data['Volume'])
            ax.set_title('Price vs Volume')
        elif plot_type == "Volume by Month":
            monthly_volume = data['Volume'].resample('M').mean()
            ax.bar(monthly_volume.index, monthly_volume.values)
            ax.set_title('Monthly Volume')
            
        self.canvas.draw()
        
    def make_prediction(self):
        try:
            data = self.load_data()
            model_name = self.model_combo.currentText()

            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]

            if model_name == "Linear Regression":
                self.train_linear_regression(train_data, test_data)
            elif model_name == "ARIMA":
                self.train_arima(train_data, test_data)
            elif model_name == "SARIMA":
                self.train_sarima(train_data, test_data)
            elif model_name == "LSTM":
                self.train_lstm(train_data, test_data)
            elif model_name == "Gradient Boosting":
                self.train_gbr(train_data, test_data)

        except Exception as e:
            self.results_label.setText(f"An error occurred: {str(e)}")

    def train_arima(self, train_data, test_data):
        try:
            model = ARIMA(train_data['Close'], order=(5, 1, 0))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test_data))
            
            mse = mean_squared_error(test_data['Close'], predictions)
            self.results_label.setText(f"ARIMA Model trained. MSE: {mse:.2f}")
            self.plot_predictions(test_data['Close'], predictions, 'ARIMA Predictions')

        except Exception as e:
            self.results_label.setText(f"ARIMA error: {str(e)}")

    def train_linear_regression(self, train_data, test_data):
        try:
            model = LinearRegression()

            # Convert the DatetimeIndex to a numerical format (ordinal numbers)
            X_train = np.array(train_data.index.map(lambda x: x.toordinal())).reshape(-1, 1)
            y_train = train_data['Close'].values
            model.fit(X_train, y_train)

            X_test = np.array(test_data.index.map(lambda x: x.toordinal())).reshape(-1, 1)
            predictions = model.predict(X_test)

            mse = mean_squared_error(test_data['Close'], predictions)
            self.results_label.setText(f"Linear Regression Model trained. MSE: {mse:.2f}")
            self.plot_predictions(test_data['Close'], predictions, 'Linear Regression Predictions')
        except Exception as e:
            self.results_label.setText(f"Linear Regression error: {str(e)}")


    def train_sarima(self, train_data, test_data):
        try:
            model = SARIMAX(train_data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            model_fit = model.fit(disp=False)
            predictions = model_fit.forecast(steps=len(test_data))
            
            mse = mean_squared_error(test_data['Close'], predictions)
            self.results_label.setText(f"SARIMA Model trained. MSE: {mse:.2f}")
            self.plot_predictions(test_data['Close'], predictions, 'SARIMA Predictions')
        except Exception as e:
            self.results_label.setText(f"SARIMA error: {str(e)}")

    def train_lstm(self, train_data, test_data):
        try:
            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train_data[['Close']])
            scaled_test = scaler.transform(test_data[['Close']])
            
            X_train, y_train = [], []
            for i in range(60, len(scaled_train)):
                X_train.append(scaled_train[i-60:i, 0])
                y_train.append(scaled_train[i, 0])
            
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
            
            X_test, y_test = [], []
            for i in range(60, len(scaled_test)):
                X_test.append(scaled_test[i-60:i, 0])
                y_test.append(scaled_test[i, 0])
            
            X_test = np.array(X_test)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            
            mse = mean_squared_error(test_data['Close'][60:], predictions[:, 0])
            self.results_label.setText(f"LSTM Model trained. MSE: {mse:.2f}")
            self.plot_predictions(test_data['Close'][60:], predictions[:, 0], 'LSTM Predictions')
        except Exception as e:
            self.results_label.setText(f"LSTM error: {str(e)}")

    def train_gbr(self, train_data, test_data):
        try:
            model = GradientBoostingRegressor()

            # Convert the DatetimeIndex to a numerical format (ordinal numbers)
            X_train = np.array(train_data.index.map(lambda x: x.toordinal())).reshape(-1, 1)
            y_train = train_data['Close'].values
            model.fit(X_train, y_train)

            X_test = np.array(test_data.index.map(lambda x: x.toordinal())).reshape(-1, 1)
            predictions = model.predict(X_test)

            mse = mean_squared_error(test_data['Close'], predictions)
            self.results_label.setText(f"Gradient Boosting Model trained. MSE: {mse:.2f}")
            self.plot_predictions(test_data['Close'], predictions, 'Gradient Boosting Predictions')
        except Exception as e:
            self.results_label.setText(f"Gradient Boosting error: {str(e)}")

    def plot_predictions(self, actual, predicted, title):
        self.pred_figure.clear()
        ax = self.pred_figure.add_subplot(111)
        ax.plot(actual.index, actual.values, label='Actual')
        ax.plot(actual.index, predicted, label='Predicted', linestyle='--')
        ax.set_title(title)
        ax.legend()
        self.pred_canvas.draw()

    def compare_models(self):
        try:
            # Load the data
            data = self.load_data()
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]

            # Store predictions and MSEs for all models
            predictions_dict = {}
            mse_dict = {}

            # Linear Regression
            model_lr = LinearRegression()
            X_train_lr = np.array(train_data.index.map(lambda x: x.toordinal())).reshape(-1, 1)
            y_train_lr = train_data['Close'].values
            model_lr.fit(X_train_lr, y_train_lr)

            X_test_lr = np.array(test_data.index.map(lambda x: x.toordinal())).reshape(-1, 1)
            predictions_lr = model_lr.predict(X_test_lr)
            mse_lr = mean_squared_error(test_data['Close'], predictions_lr)
            predictions_dict["Linear Regression"] = predictions_lr
            mse_dict["Linear Regression"] = mse_lr

            # Gradient Boosting
            model_gbr = GradientBoostingRegressor()
            model_gbr.fit(X_train_lr, y_train_lr)
            predictions_gbr = model_gbr.predict(X_test_lr)
            mse_gbr = mean_squared_error(test_data['Close'], predictions_gbr)
            predictions_dict["Gradient Boosting"] = predictions_gbr
            mse_dict["Gradient Boosting"] = mse_gbr

            # ARIMA
            model_arima = ARIMA(train_data['Close'], order=(5, 1, 0))
            model_arima_fit = model_arima.fit()
            predictions_arima = model_arima_fit.forecast(steps=len(test_data))
            mse_arima = mean_squared_error(test_data['Close'], predictions_arima)
            predictions_dict["ARIMA"] = predictions_arima
            mse_dict["ARIMA"] = mse_arima

            # SARIMA
            model_sarima = SARIMAX(train_data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            model_sarima_fit = model_sarima.fit(disp=False)
            predictions_sarima = model_sarima_fit.forecast(steps=len(test_data))
            mse_sarima = mean_squared_error(test_data['Close'], predictions_sarima)
            predictions_dict["SARIMA"] = predictions_sarima
            mse_dict["SARIMA"] = mse_sarima

            # LSTM
            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train_data[['Close']])
            scaled_test = scaler.transform(test_data[['Close']])
            X_train_lstm, y_train_lstm = [], []
            for i in range(60, len(scaled_train)):
                X_train_lstm.append(scaled_train[i-60:i, 0])
                y_train_lstm.append(scaled_train[i, 0])
            X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)
            X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))

            model_lstm = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            model_lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=1)

            X_test_lstm, y_test_lstm = [], []
            for i in range(60, len(scaled_test)):
                X_test_lstm.append(scaled_test[i-60:i, 0])
            X_test_lstm = np.array(X_test_lstm)
            X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
            predictions_lstm = model_lstm.predict(X_test_lstm)
            predictions_lstm = scaler.inverse_transform(predictions_lstm)
            mse_lstm = mean_squared_error(test_data['Close'][60:], predictions_lstm[:, 0])
            predictions_dict["LSTM"] = predictions_lstm[:, 0]
            mse_dict["LSTM"] = mse_lstm

            # Plot comparison
            self.comp_figure.clear()
            ax = self.comp_figure.add_subplot(111)
            ax.plot(test_data.index, test_data['Close'], label='Actual', color='black', linewidth=2)
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, (model, preds) in enumerate(predictions_dict.items()):
                ax.plot(test_data.index if model != "LSTM" else test_data.index[60:], preds, label=model, color=colors[i])

            ax.set_title('Model Comparison')
            ax.legend()
            self.comp_canvas.draw()

            # Display MSE results
            mse_results = "\n".join([f"{model}: MSE = {mse:.2f}" for model, mse in mse_dict.items()])
            self.results_label.setText(f"Model comparison completed.\n{mse_results}")

        except Exception as e:
            self.results_label.setText(f"Comparison error: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())