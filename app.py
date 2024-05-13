import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load pre-trained Random_Forest_search
file = 'final_model_dt.joblib'
Random_Forest_search = joblib.load(file)

# Define Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.Div([
        html.Label('Distance from Home'),
        html.Br(),
        dcc.Input(id='distance_from_home', type='number', value=0),
        html.Br(),
        html.Label('Distance from Last Transaction'),
        html.Br(),
        dcc.Input(id='distance_from_last_transaction', type='number', value=0),
        html.Br(),
        html.Label('Ratio to Median Purchase Price'),
        html.Br(),
        dcc.Input(id='ratio_to_median_purchase_price', type='number', value=0),
        html.Br(),
        html.Label('Repeat Retailer'),
        html.Br(),
        dcc.Input(id='repeat_retailer', type='number', value=0),
        html.Br(),
        html.Label('Used Chip'),
        html.Br(),
        dcc.Input(id='used_chip', type='number', value=0),
        html.Br(),
        html.Label('Used Pin Number'),
        html.Br(),
        dcc.Input(id='used_pin_number', type='number', value=0),
        html.Br(),
        html.Label('Online Order'),
        html.Br(),
        dcc.Input(id='online_order', type='number', value=0),
        html.Br(),
        html.Button('Predict', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output')
    ])
])

# Define callback to make predictions
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('distance_from_home', 'value'),
        State('distance_from_last_transaction', 'value'),
        State('ratio_to_median_purchase_price', 'value'),
        State('repeat_retailer', 'value'),
        State('used_chip', 'value'),
        State('used_pin_number', 'value'),
        State('online_order', 'value')
    ]
)
def predict_fraud(n_clicks, distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order):
    # Preprocess input data
    data = pd.DataFrame({
        'distance_from_home': [distance_from_home],
        'distance_from_last_transaction': [distance_from_last_transaction],
        'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
        'repeat_retailer': [repeat_retailer],  # Replace 0 with the actual value
        'used_chip': [used_chip],  # Replace 0 with the actual value
        'used_pin_number': [used_pin_number],  # Replace 0 with the actual value
        'online_order': [online_order]  # Replace 0 with the actual value
    })
    
    # Perform one-hot encoding on categorical columns
    categorical_columns = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
    data_encoded = pd.get_dummies(data, columns=categorical_columns)
    
    # Scale numerical columns
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded)
    
    # Make predictions
    prediction = Random_Forest_search.predict(data_scaled)
    if prediction[0] == 1:
        return "Fraud detected!"
    else:
        return "No fraud detected."


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
