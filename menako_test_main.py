import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash import dash_table
import numpy as np

# Load the DataFrame
df = pd.read_csv(r"diabetes_data.csv")
cleaned_df = pd.read_csv(r"cleaned_data.csv")


# Initialize the Dash app
app = dash.Dash(__name__)

# Divide the df to categorical variables and quantitative variables
categorical_columns = df.select_dtypes(include=['object']).columns
quantitative_columns = df.select_dtypes(include=['float64', 'int64']).columns

state_count = df['LocationAbbr'].value_counts().reset_index()
state_count.columns = ['LocationAbbr', 'Count']
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
# df['Year'] = pd.to_datetime(df['Year'], format='%Y')



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
        dcc.Slider(
            id='year-slider-1',
            min=min_year,
            max=max_year,
            value=max_year,
            marks={str(year): str(year) for year in range(min_year, max_year + 1)},
            step=1
        ),
    ]),
    dcc.Graph(id='us-heatmap-DIA01'),
    html.Div([
        dcc.Slider(
            id='year-slider-2',
            min=min_year,
            max=max_year,
            value=max_year,
            marks={str(year): str(year) for year in range(min_year, max_year + 1)},
            step=1
        ),
    ]),
    dcc.Graph(id='us-heatmap-DIA02')
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


# Callback for updating the us-heatmap for counts
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


# Callback for us-heatmap for DIA01
@app.callback(
    Output('us-heatmap-DIA01', 'figure'),
    [Input('year-slider-1', 'value')]
)
def update_us_heatmap(selected_year):
    filtered_df = df[(df['QuestionID'] == 'DIA01') &
                     (df['StratificationCategoryID1'] == 'OVERALL') &
                     (df['DataValueTypeID'] == 'CRDPREV') &
                     (df['Year'] == selected_year)]

    fig = px.choropleth(
        filtered_df,
        locations='LocationAbbr',  # State abbreviations
        locationmode="USA-states",
        color='DataValue',  # Data to be visualized
        scope="usa",
        title=f"Data for Question: \"Diabetes among adults\" in {selected_year}",
        color_continuous_scale=px.colors.sequential.Teal,
        labels={'DataValue': 'Value'}
    )
    return fig

# Callback for us-heatmap for DIA02
@app.callback(
    Output('us-heatmap-DIA02', 'figure'),
    [Input('year-slider-2', 'value')]
)
def update_us_heatmap(selected_year):
    filtered_df = df[(df['QuestionID'] == 'DIA02') &
                     (df['StratificationCategoryID1'] == 'OVERALL') &
                     (df['DataValueTypeID'] == 'CRDPREV') &
                     (df['Year'] == selected_year)]

    fig = px.choropleth(
        filtered_df,
        locations='LocationAbbr',  # State abbreviations
        locationmode="USA-states",
        color='DataValue',  # Data to be visualized
        scope="usa",
        title=f"Data for Question: \"Gestational diabetes among women with a recent live birth\" in {selected_year}",
        color_continuous_scale=px.colors.sequential.Teal,
        labels={'DataValue': 'Value'}
    )
    return fig

# # Callback for updating the correlation heatmap
# @app.callback(
#     Output('correlation-heatmap', 'figure'),
#     Input('table', 'data')
# )
# def update_heatmap(_):
#     corr = df[quantitative_columns].corr()
#     fig = px.imshow(corr, text_auto=True,
#                     labels=dict(x="Variables", y="Variables", color="Correlation"),
#                     x=quantitative_columns,
#                     y=quantitative_columns,
#                     title="Correlation Heatmap")
#     return fig


if __name__ == '__main__':
    app.run_server(debug=True)
