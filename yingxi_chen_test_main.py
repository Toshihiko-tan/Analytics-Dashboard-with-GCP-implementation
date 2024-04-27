import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from dash import dash_table
import numpy as np

# Load the DataFrame
df = pd.read_csv("diabetes_data.csv")
df = df.drop(columns = ['Unnamed: 0.1', 'TopicID'])

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
locations = df['LocationAbbr'].unique()

dia01_counts_df = dia01_counts_df = df.groupby('LocationAbbr')['QuestionID'].value_counts().reset_index()
ts_df = dia01_counts_df[dia01_counts_df['QuestionID'] == 'DIA01']

# Add a dropdown to select a categorical column for grouping in the box plot
group_dropdown = dcc.Dropdown(
    id='group-dropdown',
    options=[{'label': col, 'value': col} for col in categorical_columns],
    value=categorical_columns[0] if len(categorical_columns) > 0 else None
)



# Setup the layout of the Dash app
app.layout = html.Div([
    html.H1("Diabetes Analysis Dashboard"),
    html.H3("Dataframe Overview"),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),

    html.Div([
        # Bar chart dropdowns and chart
        html.Div([
            html.H3("Bar chart"),
            dcc.Dropdown(
                id='categorical-dropdown',
                options=[{'label': col, 'value': col} for col in categorical_columns],
                value=categorical_columns[0] if len(categorical_columns) > 0 else None
            ),
            dcc.Graph(id='bar-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        


        # Historgram of distribution of continuous variables
        html.Div([
            html.H3("Historgram"),
            dcc.Dropdown(
                id='quantitative-variable-dropdown',
                options=[{'label': i, 'value': i} for i in quantitative_columns],
                value=quantitative_columns[0]  # Default value is the first quantitative column
            ),
            dcc.Graph(id='histogram-plot')
        ], style={'width': '50%', 'display': 'inline-block'}),

    ]),


    html.Div([
    # Container Div for both plots
        html.Div([
            # DIA01 Response by State Plot and Dropdown
            html.Div([
            html.H3('DIA01 Counts by State'),
            dcc.Graph(id='dia01-count-plot'),
        ], style={'width': '50%', 'display': 'inline-block'}),

        # Box Plot Section
        html.Div([
            html.H3("Box Plot"),
            dcc.Dropdown(
                id='quantitative-dropdown',
                options=[{'label': col, 'value': col} for col in quantitative_columns],
                value=quantitative_columns[0] if len(quantitative_columns) > 0 else None
            ),
            group_dropdown, 
            dcc.Graph(id='boxplot-chart')
        ], style={'width': '50%', 'display': 'inline-block'})
    ])
]),


    html.Div([
        html.Div([
            html.H3("Diabetes Data Pair Plot"),
            dcc.Graph(
                id='pair-plot',
                figure=px.scatter_matrix(
                    df,
                    dimensions=quantitative_columns,
                    title="Pair plot of quantitative variables"
                )
            )
        ])
    ]),












    html.Div([
        html.H3("US Heatmap", style={'textAlign': 'center'}),
        dcc.Graph(id='us-heatmap')
    ]),

    html.Div([
        html.H3("US Heatmap", style={'textAlign': 'center'}),
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
        ], style={'width': '50%', 'display': 'inline-block'}),

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
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        html.H3("Pivot table", style={'textAlign': 'center'}),
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
        html.H3("Gender Bar Plot", style={'textAlign': 'center'}),
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
        html.H3("Line Chart", style={'textAlign': 'center'}),
        dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': col, 'value': col} for col in df['LocationAbbr'].unique()],
                value=locations[0], 
                clearable=False
        ),
        dcc.Graph(id='line-chart')
    ]),
])    


# Callback for updating the bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    Input('categorical-dropdown', 'value')
)

def update_bar_chart(selected_column):
    if selected_column:
        data = df[selected_column].value_counts().reset_index()
        data.columns = ['Category', 'Count']
        
        fig = px.bar(
            data,
            x='Category', 
            y='Count', 
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

#callback to update the bar plot of states in x axis
@app.callback(
    Output('dia01-count-plot', 'figure'),
    [Input('dia01-count-plot', 'id')]  # This is just a placeholder; in practice, you might have inputs that trigger this callback
)
def update_dia01_count_plot(_):
    fig = px.bar(ts_df, x='LocationAbbr', y='count', title='DIA01 Counts by State')
    return fig

# Callback for updating the histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    Input('quantitative-variable-dropdown', 'value')
)
def update_histogram(selected_variable):
    # Create the histogram using Plotly Express
    fig = px.histogram(df, x=selected_variable, nbins=30, title=f'Distribution of {selected_variable}')
    return fig


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
        title=f"Comparison of Male vs. Female Ketoacidosis Diabetes Mortality Rates by State in {selected_year}",
        labels={'value': 'Mortality Rate per 100,000', 'variable': 'Gender'},
        barmode='group',
        color_discrete_map={'SEXF': 'pink', 'SEXM': 'lightblue'}
    )
    return fig


@app.callback(
    Output('pair-plot', 'figure'),
    [Input('pairplot-variables', 'value')]
)
def update_pair_plot(selected_variables):
    # Create the SPLOM figure
    fig = px.scatter_matrix(df, dimensions=selected_variables)

    # Update layout for better readability
    fig.update_layout(
        height=1200,  # Increase figure height
        width=1200,   # Increase figure width
        margin=dict(l=50, r=50, b=50, t=50),  # Adjust margins
    )
    
    # Update axes properties
    for axis in fig.layout:
        if 'axis' in axis:
            fig.layout[axis].tickfont.size = 12  # Increase font size for ticks
            fig.layout[axis].title.font.size = 14  # Increase font size for axis titles
    
    # Rotate y-axis labels if needed
    fig.update_yaxes(tickangle=45)
    
    # Adjust marker size if needed
    fig.update_traces(marker=dict(size=5))

    return fig


# line chart
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



if __name__ == '__main__':
    app.run_server(debug=True)
