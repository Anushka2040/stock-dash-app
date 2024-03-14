import dash
import datetime
# import dash_core_components as dcc
# import dash_html_components as html
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from model import *
from datetime import date
from dash.dependencies import Input, Output, State
from dash import dcc,html

app = dash.Dash(__name__)

server = app.server


app.layout =html.Div([
        html.Div(className="bgimage",
        children=[
          html.H3("Stock Dash App", className="bgtitle"),
          html.Div(className="input" ,
            children=[
                html.P( className="start-1"),
                html.Div(className="stock-code-input", children=[
                  # stock code input
                  dcc.Input(
                  id="submit-input",
                  placeholder='',
                  type='text',
                  value=''
                  ),

                  #Submit button
                  html.Button('Submit', id='button-1')
                ]),
              
                # Date range picker input
                dcc.DatePickerRange(
                id='date-picker-range',
                start_date_placeholder_text='Start Date',
                end_date=date(2021,4,1)
                ),

                html.Div(className="buttons-2-3",children=[
                  #Stock Price button
                  html.Button('Stock Price', id='button-2'),

                  # Indicators button
                  html.Button('Indicators', id='button-3')
                ]),

                html.Div(className="forecast",children=[
                  # Number of days of forecast input
                  dcc.Input(
                  id="forecast-input",
                  placeholder='Number of days',
                  type='number',
                  value=''
                  ),

                  # Forecast button
                  html.Button('Forecast', id='button-4')
                ])
            ]),
        ]),

    html.Div(
          [
            html.Div(
                  [ # Logo
                    html.Img(id="logo-image"),
                    # Company Name
                    html.Span(id="company-description")
                  ],className='logo-name'),
            html.Div( #Description
              id="description", className="decription_ticker"),
            html.Div([
                # Stock price plot
            ], id="graphs-content"),
            html.Div([
                # Indicator plot
            ], id="main-content"),
            html.Div([
                # Forecast plot
            ], id="forecast-content")
          ],
        className="content")
],
className="container")

#Callback function for logo and company description
@app.callback(
    Output('logo-image', 'src'),
    Output('company-description','children'),
    Input('button-1','n_clicks'),
    State('submit-input','value')
    )
def get_data(n_clicks, input1):
  ticker = yf.Ticker(input1) 
  inf = ticker.info
  df = pd.DataFrame().from_dict(inf, orient="index").T
  return df["logo_url"].iloc[0],df["shortName"].iloc[0]

@app.callback(
    Output('description', 'children'),
    Input('button-1','n_clicks'),
    State('submit-input','value')
    )
def get_data(n_clicks, input1):
  ticker = yf.Ticker(input1) 
  inf = ticker.info
  df = pd.DataFrame().from_dict(inf, orient="index").T
  return df["longBusinessSummary"].iloc[0] 

#Callback function for stock price plot
@app.callback(
    Output('graphs-content', 'children'),
    Input('date-picker-range','start_date'),
    Input('date-picker-range','end_date'),
    Input('button-2','n_clicks'),
    State('submit-input','value'))
def update_data(start_date,end_date,n_clicks,input1):
  ticker = yf.Ticker(input1)
  df = yf.download( input1,start_date, end_date)
  df.reset_index(inplace=True)
  fig = get_stock_price_fig(df)
  return dcc.Graph(figure=fig)

def get_stock_price_fig(df):
  fig = px.line(df,
                x= "Date",
                y= ["Open","Close"],
                title="Closing and Opening Price vs Date")
  return fig

#Callback for indicator plot
@app.callback(
    Output('main-content', 'children'),
    Input('date-picker-range','start_date'),
    Input('date-picker-range','end_date'),
    Input('button-3','n_clicks'),
    State('submit-input','value'))
def update_data(start_date,end_date,n_clicks,input1):
  ticker = yf.Ticker(input1)
  df = yf.download( input1,start_date, end_date)
  df.reset_index(inplace=True)
  fig = get_more(df)
  return dcc.Graph(figure=fig)

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                    x= "Date",
                    y= "EWA_20",
                    title="Exponential Moving Average vs Date")

    fig.update_traces(mode= 'markers+lines')
    
    return fig

#Callback for forecast plot
@app.callback(
    Output('forecast-content', 'children'),
    Input('date-picker-range','start_date'),
    Input('date-picker-range','end_date'),
    Input('button-4','n_clicks'),
    Input('forecast-input','value'),
    State('submit-input','value'))
def update_data(start_date,end_date,n_clicks,input1,input2):
  forecast_data=pd.DataFrame()
  test_start = datetime.datetime.strptime(end_date, "%Y-%m-%d")
  dates = [test_start + pd.DateOffset(days=i) for i in range(input1)]
  forecast_data.insert(0,"Date",dates,True)
  value = forecast_indicator(start_date,end_date,n_clicks,input1,input2)
  forecast_data.insert(1,"Predicted",value,True)
  fig = px.line(forecast_data,
                    x= "Date",
                    y= "Predicted",
                    title="Predicted Prices")
  
  fig.update_traces(mode= 'markers+lines')

  return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
