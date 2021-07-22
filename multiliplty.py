import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logging import PlaceHolder
import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import seaborn as sns
from sklearn  import linear_model
from dash_html_components.Br import Br
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc



df= pd.read_csv("multilinear.csv")




rg=linear_model.LinearRegression()
#rg.fit(df[['room','bedroom','space','floor']],df.price)
#print(rg.predict([[800,120,3000,41]]))
rg.fit(df[['Company_category','Speed','Fuel_Consumption','latest']],df.Car_Price)
print(rg.coef_)
print(rg.intercept_)
print(rg.coef_[0]*800+rg.coef_[1]*120+rg.intercept_+rg.coef_[2]*300+rg.coef_[3]*41)
#print(rg.coef_[0]*800+rg.coef_[1]*120+rg.intercept_+rg.coef_[2]*300+rg.coef_[3]*41)



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout=dbc.Container([
    dbc.Row(dbc.Col(html.H1("CAR PREDICTION PROJECT",className="fio" ), width=10 )),

     html.Div([
     
       html.Label("enter Company category"),html.Br(),html.Br(),
       dcc.Input(id="c1",type="number",value=0),
  
   html.Br(),html.Br(),html.Br(),
        html.Label("enter speed"),html.Br(),
       dcc.Input(id="c2",type="number",value=0),
    html.Br(),html.Br(),
      
       html.Label("enter fuel consumption"),html.Br(),
       dcc.Input(id="c3",type="number",value=0),
    html.Br(),html.Br(),
      
       html.Label("enter latest model"),html.Br(),html.Br(),
       dcc.Input(id="c4",type="number",value=0),
   html.Br(),html.Br(),

        
     html.Div([html.H1('This is the Predicted price of the CAR'),
        html.Div(id="number-out")
    ],style={'textAlign':'center','height':'50px','color':'red','backgroundColor':'white','width':'70%',})
    



   ],className="result")


 ])

@app.callback(
    Output(component_id='number-out',component_property="children"),
    Input(component_id='c1',component_property='value'),
    Input(component_id='c2',component_property='value'),
    Input(component_id='c3',component_property='value'),
    Input(component_id='c4',component_property='value'),
)

def number_render(x, y,z,a):
    k=rg.coef_[0]*x+rg.coef_[1]*y+rg.intercept_+rg.coef_[2]*z+rg.coef_[3]*a
    return 'is: {0}'.format(k)




if __name__ == "__main__":
    app.run_server(debug=True) 