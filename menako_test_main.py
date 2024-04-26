import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash import dash_table
import numpy as np

# Load the DataFrame
df = pd.read_csv(r"diabetes_data.csv")



# Initialize the Dash app
app = dash.Dash(__name__)

# Divide the df to categorical variables and quantitative variables
categorical_columns = df.select_dtypes(include=['object']).columns
quantitative_columns = df.select_dtypes(include=['float64', 'int64']).columns

state_count = df['LocationAbbr'].value_counts().reset_index()
state_count.columns = ['LocationAbbr', 'Count']

df['Year'] = pd.to_datetime(df['Year'], format='%Y')



# Setup the layout of the Dash app
app.layout = html.Div([
    html.H1("Dataframe Overview"),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='categorical-dropdown',
                options=[{'label': col, 'value': col} for col in categorical_columns],
                value=categorical_columns[0] if len(categorical_columns) > 0 else None
            ),
            dcc.Graph(id='bar-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='quantitative-dropdown',
                options=[{'label': col, 'value': col} for col in quantitative_columns],
                value=quantitative_columns[0] if len(quantitative_columns) > 0 else None
            ),
            dcc.Graph(id='boxplot-chart')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    html.Div([
        dcc.Graph(id='us-heatmap')
    ]),
    html.Div([
        dcc.Dropdown(
            id='indicator-dropdown',
            options=[{'label': i, 'value': i} for i in df['Indicator'].unique()],
            value=df['Indicator'].unique()[0]
        ),
    ]),
    html.Div([
        dcc.Slider(
            id='year-slider',
            min=df['Year'].dt.year.min(),
            max=df['Year'].dt.year.max(),
            value=df['Year'].dt.year.min(),
            marks={str(year): str(year) for year in df['Year'].dt.year.unique()},
            step=None
        )
    ]),
    dcc.Graph(id='us-heatmap')
        html.Div([
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': col, 'value': col} for col in quantitative_columns],
                value=quantitative_columns[0] if len(quantitative_columns) > 0 else None
            ),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': col, 'value': col} for col in quantitative_columns],
                value=quantitative_columns[1] if len(quantitative_columns) > 1 else quantitative_columns[0]
            ),
            dcc.Graph(id='regression-plot')
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='correlation-heatmap')
        ], style={'width': '50%', 'display': 'inline-block'})
    ])
])


# Callback for updating the bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    Input('categorical-dropdown', 'value')
)
def update_bar_chart(selected_column):
    if selected_column:
        fig = px.bar(df, x=selected_column, title=f"Counts of {selected_column}", color_discrete_sequence=['#636EFA'])
        return fig
    return {}

# Callback for updating the boxplot
@app.callback(
    Output('boxplot-chart', 'figure'),
    Input('quantitative-dropdown', 'value')
)
def update_boxplot(selected_column):
    if selected_column:
        fig = px.box(df, y=selected_column, title=f"Distribution of {selected_column}", notched=True, color_discrete_sequence=['#636EFA'])
        return fig
    return {}

@app.callback(
    Output('us-heatmap', 'figure'),
    Input('table', 'data')
)
def update_us_heatmap(_):
    fig = px.choropleth(
        state_count, 
        locations='LocationAbbr', 
        locationmode="USA-states", 
        color='Count',  # This should be the column from your DataFrame that holds the counts
        scope="usa",
        title="Value Counts by State"
    )
    return fig

@app.callback(
    Output('us-heatmap', 'figure'),
    [Input('indicator-dropdown', 'value'), Input('year-slider', 'value')]
)
def update_us_heatmap(selected_indicator, selected_year):
    filtered_df = df[(df['Indicator'] == selected_indicator) &
                     (df['Year'].dt.year == selected_year)]
    
    fig = px.choropleth(
        filtered_df,
        locations='LocationAbbr',  # This column should have state abbreviations
        locationmode="USA-states",
        color="Value",  # This should be the column with the data you want to visualize
        scope="usa",
        title=f"{selected_indicator} by State in {selected_year}",
        color_continuous_scale=px.colors.sequential.Teal,
        labels={'Value': 'Indicator Value'}
    )
    fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    return fig

# Callback for updating the regression plot
@app.callback(
    Output('regression-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'), Input('y-axis-dropdown', 'value')]
)
def update_regression_plot(x_col, y_col):
    if x_col and y_col:
        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"Regression between {x_col} and {y_col}")
        return fig
    return {}

# Callback for updating the correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('table', 'data')
)
def update_heatmap(_):
    corr = df[quantitative_columns].corr()
    fig = px.imshow(corr, text_auto=True,
                    labels=dict(x="Variables", y="Variables", color="Correlation"),
                    x=quantitative_columns,
                    y=quantitative_columns,
                    title="Correlation Heatmap")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
