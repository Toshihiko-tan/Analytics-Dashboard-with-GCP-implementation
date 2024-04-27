import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from dash import dash_table
import numpy as np
import pickle

# Load the DataFrame
df = pd.read_csv(r"diabetes_data.csv")
df = df.drop(columns = ['Unnamed: 0.1', 'TopicID'])
cleaned_data = pd.read_csv('cleaned_data.csv')


# Initialize the Dash app
app = dash.Dash(__name__)

# Divide the df to categorical variables and quantitative variables
categorical_columns = df.select_dtypes(include=['object']).columns
quantitative_columns = df.select_dtypes(include=['float64', 'int64']).columns

state_count = df['LocationAbbr'].value_counts().reset_index()
state_count.columns = ['LocationAbbr', 'Count']
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
min_year_clean = int(cleaned_data['YearStart'].min())
# df['Year'] = pd.to_datetime(df['Year'], format='%Y')


# Add a dropdown to select a categorical column for grouping in the box plot
group_dropdown = dcc.Dropdown(
    id='group-dropdown',
    options=[{'label': col, 'value': col} for col in categorical_columns],
    value=categorical_columns[0] if len(categorical_columns) > 0 else None
)

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
        # Bar chart dropdowns and chart
        html.Div([
            dcc.Dropdown(
                id='categorical-dropdown',
                options=[{'label': col, 'value': col} for col in categorical_columns],
                value=categorical_columns[0] if len(categorical_columns) > 0 else None
            ),
            dcc.Graph(id='bar-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        # Box plot dropdowns and chart
        html.Div([
            dcc.Dropdown(
                id='quantitative-dropdown',
                options=[{'label': col, 'value': col} for col in quantitative_columns],
                value=quantitative_columns[0] if len(quantitative_columns) > 0 else None
            ),
            group_dropdown,  # New dropdown for selecting a group for the box plot
            dcc.Graph(id='boxplot-chart')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),

    html.Div([
        dcc.Graph(id='us-heatmap')
    ]),

    html.Div([
        html.Div([
            dcc.Slider(
                id='year-slider-1',
                min=min_year,
                max=max_year,
                value=max_year,
                marks={str(year): str(year) for year in range(min_year, max_year + 1)},
                step=1
            ),
            dcc.Graph(id='us-heatmap-DIA01')
        ]),

        html.Div([
            dcc.Slider(
                id='year-slider-2',
                min=min_year,
                max=2021,
                value=2021,
                marks={str(year): str(year) for year in range(min_year, 2021 + 1)},
                step=1
            ),
            dcc.Graph(id='us-heatmap-DIA02')
        ])
    ]),
    
    html.Div([
        dcc.Slider(
            id='year-slider-3',
            min=min_year,
            max=2021,
            value=2021,
            marks={str(year): str(year) for year in range(min_year, 2021 + 1)},
            step=1
        ),
        dcc.Graph(id='pivot_table'),
    ]),

    html.Div([
        dcc.Slider(
            id='year-slider-4',
            min=min_year,
            max=2021,
            value=2021,
            marks={str(year): str(year) for year in range(min_year, 2021+1)},
            step=1
        ),
        dcc.Graph(id='gender-bar-plot'),
    ]),

    html.Div([
        dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': col, 'value': col} for col in df['LocationAbbr'].unique()],
        ),
        dcc.Graph(id='line-chart')
    ]),

    html.Div([
        dcc.Graph(id='correlation-heatmap')
    ])
])    


# Callback for updating the bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    Input('categorical-dropdown', 'value')
)
def update_bar_chart(selected_column):
    if selected_column:
        fig = px.bar(
            df[selected_column].value_counts().reset_index(),
            x='index',
            y=selected_column,
            title=f"Counts of {selected_column}"
        )
        return fig
    return {}

# Callback for updating the boxplot
@app.callback(
    Output('boxplot-chart', 'figure'),
    [Input('quantitative-dropdown', 'value'),
     Input('group-dropdown', 'value')]  # Take input from the new group dropdown as well
)
def update_boxplot(selected_quantitative, selected_group):
    if selected_quantitative and selected_group:
        fig = px.box(df, y=selected_quantitative, x=selected_group, 
                     title=f"Distribution of {selected_quantitative} by {selected_group}", notched=True)
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
    filtered_df = cleaned_data[(cleaned_data['QuestionID'] == 'DIA01') &
                     (cleaned_data['StratificationCategoryID1'] == 'OVERALL') &
                     (cleaned_data['DataValueTypeID'] == 'CRDPREV') &
                     (cleaned_data['YearStart'] == selected_year)]

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

# Callback for state and race
@app.callback(
    Output('pivot_table', 'figure'),
    Input('year-slider-3', 'value')
)
def state_race(selected_year):
    filtered_data = df[(df['QuestionID'] == 'DIA04') &
                    (df['StratificationCategoryID1'] == 'RACE') &
                    (df['DataValueTypeID'] == 'CRDRATE') &
                    (df['Year'] == selected_year)]
    pivot_data = filtered_data.pivot_table(
    values='DataValue',
    index='StratificationID1', 
    columns='LocationAbbr', 
    )

    fig = px.imshow(
        pivot_data,
        labels=dict(x="State", y="Race", color="Mortality Rate"),
        x=pivot_data.columns,
        y=pivot_data.index,
        title=f"Diabetes Ketoacidosis Mortality Rate per 100,000 by State and Race in {selected_year}",
        aspect="auto",
        color_continuous_scale='Blues'
    )

    # Adding text to each cell manually
    fig.update_traces(texttemplate="%{z:.1f}", textfont={"size": 10})
    return fig

# Create gender comparison bar plot
@app.callback(
    Output('gender-bar-plot', 'figure'),
    Input('year-slider-4', 'value')
)

def gender_bar_plot(selected_year):
    gender_filtered_data = df[
        (df['QuestionID'] == 'DIA04') &
        (df['Year'] == selected_year) &
        (df['StratificationCategoryID1'] == 'SEX') &
        (df['DataValueTypeID'] == 'CRDRATE')
    ]
    pivot_gender_data = gender_filtered_data.pivot_table(
        values='DataValue',
        index='LocationAbbr',
        columns='StratificationID1',
    )
    # Create a bar plot using Plotly to compare male vs. female diabetes mortality rates by state
    fig = px.bar(
        pivot_gender_data.reset_index(),
        x='LocationAbbr',
        y=['SEXF', 'SEXM'],
        title=f"Comparison of Sex Ketoacidosis Diabetes Mortality Rates by State in {selected_year}",
        labels={'value': 'Mortality Rate per 100,000', 'variable': 'Gender'},
        barmode='group',
        color_discrete_map={'SEXF': 'pink', 'SEXM': 'lightblue'}
    )
    return fig

@app.callback(
    Output('line-chart', 'figure'),
    Input('state-dropdown', 'value')
)

def update_line_chart(selected_state):
    # Create a scatter trace
    mortality_AL = df[(df['QuestionID'] == 'DIA03') & 
                            (df['StratificationCategoryID1'] == 'OVERALL') &
                            (df['DataValueTypeID'] == 'CRDRATE') &
                            (df['LocationAbbr'] == selected_state) 
                            ][['Year', 'DataValue']]
    trace = go.Scatter(
        x=mortality_AL['Year'],
        y=mortality_AL['DataValue'],
        mode='lines+markers',
        name='Mortality Rate for AL'
    )
    
    # Create the figure
    fig = go.Figure(data=[trace])

    # Update layout
    fig.update_layout(
        title=f'Mortality Rate for {selected_state} (2019-2021)',
        xaxis_title='Year',
        yaxis_title='DataValue',
        xaxis=dict(
            tickmode='array',
            tickvals=mortality_AL['Year'].unique()
        )    
    )
    
    return fig

# Callback for the correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('table', 'data')]
)
def update_correlation_heatmap(selected_year):
    dia01 = cleaned_data[(cleaned_data['QuestionID'] == 'DIA01') 
                     & (cleaned_data['StratificationCategoryID1'] == 'OVERALL') 
                     & (cleaned_data['DataValueTypeID'] == 'CRDPREV') 
                    #  & (cleaned_data['LocationAbbr'] == 'AL')
                     & (cleaned_data['YearStart'] == 2019)
                     ][['LocationAbbr', 'DataValue']]
    dia02 = cleaned_data[(cleaned_data['QuestionID'] == 'DIA02') 
                         & (cleaned_data['StratificationCategoryID1'] == 'OVERALL') 
                         & (cleaned_data['DataValueTypeID'] == 'CRDPREV') 
                        #  & (cleaned_data['LocationAbbr'] == 'AL')
                         & (cleaned_data['YearStart'] == 2019)
                         ][['LocationAbbr', 'DataValue']]
    dia03 = cleaned_data[(cleaned_data['QuestionID'] == 'DIA03')
                         & (cleaned_data['StratificationCategoryID1'] == 'OVERALL')
                         & (cleaned_data['DataValueTypeID'] == 'CRDRATE') 
                        #  & (cleaned_data['LocationAbbr'] == 'AL')
                         & (cleaned_data['YearStart'] == 2019) 
                         ][['LocationAbbr', 'DataValue']]
    dia04 = cleaned_data[(cleaned_data['QuestionID'] == 'DIA04')
                         & (cleaned_data['StratificationCategoryID1'] == 'OVERALL')
                         & (cleaned_data['DataValueTypeID'] == 'CRDRATE') 
                        #  & (cleaned_data['LocationAbbr'] == 'AL')
                         & (cleaned_data['YearStart'] == 2019) 
                         ][['LocationAbbr', 'DataValue']]
    final_df = pd.merge(dia01, dia02, on='LocationAbbr', suffixes=('_dia01', '_dia02'))
    final_df = pd.merge(final_df, dia03, on='LocationAbbr')
    final_df = pd.merge(final_df, dia04, on='LocationAbbr', suffixes=('_dia03', '_dia04'))

    correlation_matrix = final_df[['DataValue_dia01', 'DataValue_dia02', 'DataValue_dia03', 'DataValue_dia04']].corr()

    fig = go.Figure(data=go.Heatmap(
        z=np.array(correlation_matrix),
        x=correlation_matrix.columns,
        y=correlation_matrix.index,\
        colorscale='reds',
        text=np.around(correlation_matrix.to_numpy(), decimals=2),  # round the correlation values to 2 decimals
        texttemplate="%{text}"))

    # Update layout
    fig.update_layout(
        title='Correlation Heatmap between questions',
        xaxis=dict(title='Variables',
                   tickmode='array',
                   tickvals=[0,1,2,3],
                   ticktext=['DIA01', 'DIA02', 'DIA03', 'DIA04']),
        yaxis=dict(title='Variables',
                   tickmode='array',
                   tickvals=[0,1,2,3],
                   ticktext=['DIA01', 'DIA02', 'DIA03', 'DIA04']),
        height=500,  # Adjust the height to your preference
        width=500   # Adjust the width to your preference
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
