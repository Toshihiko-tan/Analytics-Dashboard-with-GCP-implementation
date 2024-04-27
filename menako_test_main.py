import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from dash import dash_table
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the DataFrame
df = pd.read_csv(r"diabetes_data.csv")
df = df.drop(columns = ['Unnamed: 0.1', 'TopicID'])
cleaned_data = pd.read_csv('cleaned_data.csv')
final_df = pd.read_csv(r"final_df.csv")


# Initialize the Dash app
app = dash.Dash(__name__)

# Divide the df to categorical variables and quantitative variables
categorical_columns = df.select_dtypes(include=['object']).columns
quantitative_columns = df.select_dtypes(include=['float64', 'int64']).columns

ts_df = df[df['QuestionID'] == 'DIA01'].groupby('LocationAbbr').size().reset_index(name='count')


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
    html.H1("Diabetes Analysis Dashboard", style={'textAlign': 'center'}),
    html.H2("Dataframe Overview", style={'textAlign': 'center'}),
    html.P("Our goal is to predict the percentage of diabetes patients among adults using different predictors."),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),

    html.Div([
        # Bar chart dropdowns and chart
        html.Div([
            html.H2("Bar chart"),
            dcc.Dropdown(
                id='categorical-dropdown',
                options=[{'label': col, 'value': col} for col in categorical_columns],
                value=categorical_columns[0] if len(categorical_columns) > 0 else None
            ),
            dcc.Graph(id='bar-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.H2("Historgram"),
            dcc.Dropdown(
                id='quantitative-variable-dropdown',
                options=[{'label': i, 'value': i} for i in quantitative_columns],
                value=quantitative_columns[0]  # Default value is the first quantitative column
            ),
            dcc.Graph(id='histogram-plot')
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),

    html.Div([
        html.H2("Separate Bar Chart for Different Questions"),
        html.Div([dcc.Graph(id='bar-chart-dia01')], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bar-chart-dia02')], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bar-chart-dia03')], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bar-chart-dia04')], style={'width': '25%', 'display': 'inline-block'}),
    ]),

    html.Div([    
        # Box plot dropdowns and chart
        html.H2("Box Plot"),
        dcc.Dropdown(
            id='quantitative-dropdown',
            options=[{'label': col, 'value': col} for col in quantitative_columns],
            value=quantitative_columns[0] if len(quantitative_columns) > 0 else None
        ),
        group_dropdown,  # New dropdown for selecting a group for the box plot
        dcc.Graph(id='boxplot-chart')
    ]),

    html.Div([
        html.Div([
            html.H2("Diabetes Data Pair Plot for Different Questions"),
            dcc.Graph(id='pair-plot')
        ])
    ]),

    html.Div([
        html.H2("Counts for Each States"),
        html.P("We want to examine the number of data points we have for each state. Our assumption is that the number of data points differ by states, where states with higher population, like New York, California, and Texas have more data points than others."),
        html.P("We have created a heatmap to visualize the data points we have, with colors that are more yellow have more data points than others."),
        html.Div([
            dcc.Graph(id='us-heatmap')
        ]),
        html.P("Indeed, we can see from the heatmap that states with higher population, New York, Washington, California, and Texas have the most data points. Wyoming, South Dakota, North Dakota, Idaho, and Montana are agriculture states with less population, and hence have less data points than other states."),
    ]),
    
    html.H2("Data Value of DIA01 for Each State in Different Years"),
    html.P("Then, we want to visualize our response variable, DIA01, or precentage of diabete patients among adults. Our assumption is that the percentage of diabetes among adults varies by states, but not by time."),
    html.P("Again, we have created a heatmap with sliders to select the data year we have. For our response variable, we have four years of data, from 2019 to 2022."),
    
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
        html.P("Immediately, we discovered that 2020 and 2021 have the exact same data. After using the original dataset to verify that this is indeed the case, we believe that this is due to government cannot collect data from people during the covid-19 pandemic, so the data was used for the consecutive two years. Other than that, we can see that there does to seem some effect of year on the percentage of diabetes patients among adults. Specifically, the year 2022 has generally lower diabete rates across all states than previous years."),
        html.P("Also, we can see that there seems to be some significance relationship between state and percentage of diabete patients among adults. Specifically, southern states seem to have higher percentage of diabetes patients among adults than any other regions, but West Virginia has the highest percentage of diabetes patients among adults than any other states."),
        
        html.H2("Data Value of DIA02 for Each State in Different Years"),

        html.P("We have also visualized the percentage of Gestational diabetes among women with a recent live birth using a heatmap. Our assumption is that, again, this would differ much by states, and would have similar distribution to our response variable."),
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
        ]),
        html.P("From the heatmap, we can immediately see that there are many missing values for multiple states. Specifically, California, Texas, and Nevada are missing all three years of data from 2019 to 2021. These are some of the most populated states in Amercica, so we should definitely take that into consideration."),
        html.P("We can see that the distribution of this variable, in fact, is quite different from our reponse variable, percentage of diabetes patients in adults. States like New York, South Dakota, Washington, and Oregon, which don't have very high percentage of diabetes patients in adults, have much higher percentage of Gestational diabetes among women with a recent live birth. This is quite different from our assumption, and it requires further inspection when we fit the machine learning model."),
    ]),
    
    html.H2("Diabetes Ketoacidosis Mortality Rate by State and Race in Different Years"),
    
    html.Div([
        html.P("Then, we visualized another predictor variable, Diabetes Ketoacidosis Mortality Rate (per 100,000). We plot a pivottable by state and race. Our assumption is that Diabetes Ketoacidosis Mortality Rate will differ more by states and not much by race."),
        dcc.Slider(
            id='year-slider-3',
            min=min_year,
            max=2021,
            value=2021,
            marks={str(year): str(year) for year in range(min_year, 2021 + 1)},
            step=1
        ),
        dcc.Graph(id='pivot_table'),
        html.P("Again, we can immediately see that the data is missing for most races except white and black. There is, however, valid data for all of the states. Therefore, we have to consider how representative this data is of the diverse ethnicities in America. We can immediately see that native Americans of Oklahoma, a state where native Americans reside traditionally, have the highest Diabetes Ketoacidosis Mortality Rate (per 100,000) across the years where the data is available. While other states generally have similar mortality rates, Kentucky and Nevada are two states that constantly have higher Diabetes Ketoacidosis Mortality Rate for both white and black demographics."),
    ]),

    html.H2("Diabetes Ketoacidosis Mortality Rate against Sex in Different Years"),

    html.Div([
        html.P("We have also plot the Diabetes Ketoacidosis Mortality Rate (per 100,000) against sex by state. Our assumption is that the Diabetes Ketoacidosis Mortality Rate is affected by sex."),
        dcc.Slider(
            id='year-slider-4',
            min=min_year,
            max=2021,
            value=2021,
            marks={str(year): str(year) for year in range(min_year, 2021+1)},
            step=1
        ),
        dcc.Graph(id='gender-bar-plot'),
        html.P("Indeed, our assumption is true, and we can see that the Diabetes Ketoacidosis Mortality Rate for male is higher in almost all the states across every year which data is available. In fact, in states like New Mexico, Nevada, and Oklohama, the Diabetes Ketoacidosis Mortality Rate for male is much higher than that of female's mortality rate."),
    ]),

    html.Div(style={'marginBottom': '50px'}),

    html.Div([
        html.P("We have also plotted the predictor variable Mortality Rate of diabetes with a line plot with data from 2019 to 2021. Our assumption is that it does change with time."),
        dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': col, 'value': col} for col in df['LocationAbbr'].unique()],
        ),
        dcc.Graph(id='line-chart'),
        html.P("After examining the line plot or all the states, we can see that for most of the states, the Mortality Rate of diabetes do increase with time from 2019 to 2021, as the mortality rate of 2021 is generally higher than the mortality rate of 2019. There are, however, some states, like New York, that have a decreasing diabetes mortality rate from 2020 to 2021."),
    ]),

    html.H2("Correlation Heatmap between Different Questions"),

    html.Div([
        html.P("Lastly, we plot the correlation heatmap of our predictor and response variable. Our assumption is that there are some correlation between the predictor variables, due to the nature of the predictor variables."),
        dcc.Graph(id='correlation-heatmap',style={'margin-left': 'auto', 'margin-right': 'auto', 'width': '50%'}),
        html.P("The result is actually not quite what we were expecting. There doesn't seem to be high correlation between the predicting variables, so multicollinearity might not be as big a problem as we thought. The only pair we might need to keep an eye on are DIA03 and DIA04, further diagnosis might be needed in model selection process.")
    ]),


    html.H1("Predictive Analysis Using ML Models", style={'textAlign': 'center'}),
    html.H2("Model Selection", style={'textAlign': 'center'}),
    html.P("To investigate the prevalence of diabetes among adults, we selected \"DataValue_dia01\" as our response variable. For the selection of predictors, we first identified potential relevant variables from the dataset that are associated with diabetes prevalence, as well as those of particular interest. Through the use of a correlation matrix, we analyzed the relationships between variables and selected \"DataValue_dia02\", \"DataValue_dia03\", \"DataValue_dia04\", \"LocationAbbr\", and \"YearStart\" as our predictors. Subsequent checks with Variance Inflation Factor (VIF) confirmed that there was no significant multicollinearity among these predictors."),
    html.P("We tested six different regression models: Linear Regression, Random Forest, Ridge Regression, Lasso Regression, Support Vector Machine (SVM), and Decision Tree. The models were evaluated using two metrics: R-square and Mean Squared Error (MSE)."),
    html.Div([
        dcc.Graph(id='MSE'),
        dcc.Graph(id='R-squared'),
    ]),
    html.P("Our analysis revealed that Linear Regression performed the best in terms of achieving the highest R-square and the lowest MSE. While both Random Forest and Ridge Regression also showed strong performances, Linear Regression was preferred due to its superior interpretability."),

    html.H2("Prediction Using Linear Regression Model", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Div([
                html.Label('State:', style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='state-dropdown-linear',
                    options=[{'label': col, 'value': col} for col in df['LocationAbbr'].unique()],
                    style={'width': '300px'}  # Adjust the width as needed
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),

            html.Div([
                html.Label('Year:', style={'margin-right': '10px'}),
                dcc.Input(id='year-linear', type='number', placeholder='Enter Year')
            ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),

            html.Div([
                html.Label('DIA02:', style={'margin-right': '10px'}),
                dcc.Input(id='dia02-linear', type='number', placeholder='Enter DIA02')
            ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
            html.Div([
                html.Label('DIA03:', style={'margin-right': '10px'}),
                dcc.Input(id='dia03-linear', type='number', placeholder='Enter DIA03')
            ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),

            html.Div([
                html.Label('DIA04:', style={'margin-right': '10px'}),
                dcc.Input(id='dia04-linear', type='number', placeholder='Enter DIA04')
            ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),

            html.Button('Predict!!', id='predict-button-linear', n_clicks=0),
        ], style={'display': 'flex', 'flexDirection': 'column', 'width': '50%', 'marginLeft': 'auto', 'marginRight': 'auto'}),
    
    html.Div(id='prediction-output-linear', style={'textAlign': 'center'})
    ]),

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



# Callback for updating the histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    Input('quantitative-variable-dropdown', 'value')
)
def update_histogram(selected_variable):
    # Create the histogram using Plotly Express
    fig = px.histogram(df, x=selected_variable, nbins=30, title=f'Distribution of {selected_variable}')
    return fig

def create_bar_chart_for_dia(dia_number):
    # Filter the DataFrame for the specified DIA number
    filtered_df = df[df['QuestionID'] == dia_number]
    # Aggregate by state
    aggregated_data = filtered_df.groupby('LocationAbbr').size().reset_index(name='Count')
    # Create the bar chart
    fig = px.bar(aggregated_data, x='LocationAbbr', y='Count', title=f'Counts for {dia_number}')
    return fig


# Callbacks for updating each bar chart
@app.callback(Output('bar-chart-dia01', 'figure'), [Input('categorical-dropdown', 'value')])
def update_bar_chart_dia01(_):
    return create_bar_chart_for_dia('DIA01')

@app.callback(Output('bar-chart-dia02', 'figure'), [Input('categorical-dropdown', 'value')])
def update_bar_chart_dia02(_):
    return create_bar_chart_for_dia('DIA02')

@app.callback(Output('bar-chart-dia03', 'figure'), [Input('categorical-dropdown', 'value')])
def update_bar_chart_dia03(_):
    return create_bar_chart_for_dia('DIA03')

@app.callback(Output('bar-chart-dia04', 'figure'), [Input('categorical-dropdown', 'value')])
def update_bar_chart_dia04(_):
    return create_bar_chart_for_dia('DIA04')


@app.callback(
    Output('pair-plot', 'figure'),
    Input('table', 'data')
)    

def update_pair_plot(_):
    fig=px.scatter_matrix(
        final_df,
        dimensions=['DataValue_dia01','DataValue_dia02','DataValue_dia03','DataValue_dia04'],
        title="Pair plot for different questions"
    )
    axis_titles = {
    'DataValue_dia01': 'DIA01',
    'DataValue_dia02':'DIA02',
    'DataValue_dia03':'DIA03',
    'DataValue_dia04':'DIA04'
    }

    # Update x and y axis labels
    for i, dim in enumerate(fig.data[0].dimensions):
        dim.label = axis_titles[dim.label]
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

# Callback for MSE and R-squared Graph
@app.callback(
    Output('MSE', 'figure'),
    [Input('table', 'data')]
)
def update_mse_graph(_):
    mse_scores = {
    'Linear Regression': 0.5154938,
    'Random Forest': 3.237268,
    'Ridge': 1.167645,
    'Lasso': 4.613817,
    'SVR': 4.90765,
    'Decision Tree': 7.4525
    }
    fig = go.Figure([go.Bar(x=list(mse_scores.keys()), y=list(mse_scores.values()), marker_color='lightblue')])

    fig.update_layout(
        title='Comparison of MSE Across Different Regression Models',
        xaxis_title='Model Type',
        yaxis_title='MSE',
        template='plotly_white'  # Change template as needed
    )
    return fig

@app.callback(
    Output('R-squared', 'figure'),
    [Input('table', 'data')]
)
def update_r_squared_graph(_):
    r2_scores = {
    'Linear Regression': 0.983583,
    'Random Forest': 0.922726,
    'Ridge': 0.9148073,
    'Lasso': 0.2311491,
    'SVR': 0.003719412,
    'Decision Tree': 1
    }
    fig = go.Figure([go.Bar(x=list(r2_scores.keys()), y=list(r2_scores.values()), marker_color='pink')])

    fig.update_layout(
        title='Comparison of R^2 Scores Across Different Regression Models',
        xaxis_title='Model Type',
        yaxis_title='R^2 Score',
        template='plotly_white'  # Change template as needed
    )
    return fig


# Callback for the prediction using linear model
@app.callback(
    Output('prediction-output-linear', 'children'),
    [Input('predict-button-linear', 'n_clicks')],
    [State('state-dropdown-linear', 'value'),
     State('year-linear', 'value'),
     State('dia02-linear', 'value'),
     State('dia03-linear', 'value'),
     State('dia04-linear', 'value')]
)
def predict(n_clicks, location, year, dia02, dia03, dia04):
    with open('linear_model.pkl', 'rb') as file:
        model = pickle.load(file)
    if n_clicks > 0:
        # Create an input DataFrame or array depending on your model's expected input
        input_features = [location, year, dia02, dia03, dia04]
        
        # Assuming the model expects a DataFrame with specific column names
        test_data = pd.DataFrame([input_features], 
                    columns=['LocationAbbr','YearStart', 'DataValue_dia02', 'DataValue_dia03', 'DataValue_dia04'])
        
        columns_to_encode = ['LocationAbbr']

        # Perform one-hot encoding
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')
        merged_df = pd.read_csv("merged_df.csv")
        x_train = merged_df.drop(['DataValue_dia01'], axis=1)
        x_train_encoded = ct.fit_transform(x_train)
        test_data_encoded = ct.transform(test_data)
        # Make prediction
        prediction = model.predict(test_data_encoded)
        return f'Predicted DIA01: {prediction[0]}'

    return 'Enter values and press predict.'


if __name__ == '__main__':
    app.run_server(debug=True)
