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
from data_cleaning_module import diabete_data
import visualization_and_prediction_module as vp
import utility_module as um

# Load the DataFrame
df = diabete_data.drop(columns = ['Unnamed: 0.1', 'TopicID', 'Unnamed: 0'])
cleaned_data = pd.read_csv('Data/cleaned_data.csv')
final_df = pd.read_csv(r"Data/final_df.csv")


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
    html.P("Our goal is to predict the percentage of diabetes patients among adults using different predictors and identify the most important predictors. We will first explore the dataset to understand the distribution of the data and the relationship between different variables. Then, we will use machine learning models to predict the percentage of diabetes patients among adults. We will also analyze the importance of different predictors in predicting the percentage of diabetes patients among adults."),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),

    html.Div([
        # Bar chart dropdowns and chart
        html.P("First, we want to display some basic histogram and barplots from the dataset."),
        html.Div([
            html.H2("Bar chart"),
            dcc.Dropdown(
                id='categorical-dropdown',
                options=[{'label': col, 'value': col} for col in categorical_columns],
                value=categorical_columns[0] if len(categorical_columns) > 0 else None
            ),
            dcc.Graph(id='bar-chart'),
            html.P("For the bar chart of QuestionID, we can see the count of questions representing our predictor variables: DIA02 (Gestational diabetes among women with a recent live birth), DIA03 (Diabetes mortality among all people, underlying or contributing cause), DIA04 (Diabetic ketoacidosis mortality among all people, underlying or contributing cause), and our response variable DIA01 (Diabetes among adults). We can see that DIA03 has the highest count, followed closely by DIA01 and DIA04, while DIA02 has the lowest count."),
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.H2("Histogram"),
            dcc.Dropdown(
                id='quantitative-variable-dropdown',
                options=[{'label': i, 'value': i} for i in quantitative_columns],
                value=quantitative_columns[0]  # Default value is the first quantitative column
            ),
            dcc.Graph(id='histogram-plot'),
            html.P("For the histogram, we can examine the distribution of the data values and low/high confidence limit. Most of the variables displayed here seem to be heavily skewed to the right."),
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),

    html.Div([
        html.H2("Separate Bar Chart for Different Questions"),
        html.P("Then, we want to examine the count of different questions by states. Considering that the data points of each state are most likely determined by population size and how active the research of diabetes are in these states, we assumed that the distribution of count of questions by states wouldn't differ too much."),
        html.Div([dcc.Graph(id='bar-chart-dia01')], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bar-chart-dia02')], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bar-chart-dia03')], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bar-chart-dia04')], style={'width': '25%', 'display': 'inline-block'}),
        html.P("However, the bar chart suggests that the distribution of count of questions by states do differ from questions and questions. While states like California, New York, and Washington mostly have high counts in all four questions, the other states vary quite a lot switching from question to question."),
    ]),

    html.Div([    
        # Box plot dropdowns and chart
        html.H2("Box Plot"),
        html.P("The Boxplot is great for displaying the relationship between different variables and the year they were recorded in. Our assumption is that all the data are conducted in a similar timeframe."),
        dcc.Dropdown(
            id='quantitative-dropdown',
            options=[{'label': col, 'value': col} for col in quantitative_columns],
            value=quantitative_columns[0] if len(quantitative_columns) > 0 else None
        ),
        group_dropdown,  # New dropdown for selecting a group for the box plot
        dcc.Graph(id='boxplot-chart'),
        html.P("Indeed, after examine different boxplots between year and multiple other features, like the location where the survey took place, we can confirm that indeed, most of the data are collected in a similar timeframe."),
    ]),

    html.Div([
        html.Div([
            html.H2("Diabetes Data Pair Plot for Different Questions"),
            html.P("Then, we want to explore the correlation between predictor and response variables. Our assumption is that there is some correlation between the predictor variables, due to the way how these survey questions are structured, and that they are all about the topic of diabetes."),
            dcc.Graph(id='pair-plot'),
            html.P("For the predictor variables, we can see from the pair plots that there doesn't seem to be very high correlation between DIA04 and DIA01, DIA02, DIA03. There seems to be some weak positive correlation between DIA02 and DIA03, as well as DIA02 and DIA04. So there indeed seems to be some positive correlation between the predictor variables, like we assumed. For our response variable, We can see that DIA01 seems to have weak positive correlation betwen all DIA02, DIA03, and DIA04."),
        ])
    ]),

    html.Div([
        html.H2("Counts for Each States"),
        html.P("Then, we want to examine the total number of data points we have for each state, regardless the question. Our assumption is that the number of data points differ by states, where states with higher population, like New York, California, and Texas have more data points than others."),
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


    html.Div([
        html.H2("Mortality Rate Trends", style={'textAlign': 'center'}),
        html.P("We have also plotted the predictor variables and the response variable with a line plot with data from 2019 to 2021 to show the trend. Our assumption is that it does change with time."),
        dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': col, 'value': col} for col in df['LocationAbbr'].unique()],
            value=df['LocationAbbr'].unique()[0], 
            clearable=False
        ),
        html.Div([
            dcc.Graph(id='trend-chart-dia01')
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='trend-chart-dia02')
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='trend-chart-dia03')
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='trend-chart-dia04')
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.P("After examining the line plot or all the states, we can see that for most of the states, the trend for predictor and response variables are to increase with time from 2019 to 2021. For example, the mortality rate of 2021 is higher than the mortality rate of 2019 for most of the states. There are, however, some states, like New York, that have a decreasing diabetes mortality rate from 2020 to 2021.")
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
    
    html.Div([
        html.H2("Conclusion & Final Thoughts", style={'textAlign': 'center'}),
        html.P("""
               We explored six regression models: Linear Regression, Random Forest, Ridge Regression, Lasso Regression, Support Vector Machine (SVM), and Decision Tree. We utilized R-square and Mean Squared Error (MSE) to evaluate these models. Due to the limited data availability for certain states, we encountered challenges in employing cross-validation, which could potentially leave some state data untrained if allocated to test data. To demonstrate our evaluation approach, we manually selected four data points as test data, using the remainder for training, which provided us with an estimation of MSE.

            To explore the prevalence of diabetes across different states, we focused on "DataValue_dia01," which represents the percentage of adults diagnosed with diabetes. This variable served as our response variable in the analysis. To identify relevant predictors, we meticulously selected variables from the dataset that potentially correlate with the diabetes prevalence, and also incorporated variables that specifically interested us, such as variables representing the year and region.
Utilizing a correlation matrix, we analyzed the relationships between variables and selected "DataValue_dia02," "DataValue_dia03," "DataValue_dia04," "LocationAbbr," and "YearStart" as our predictors. These variables are particularly significant as:
"DataValue_dia02" represents the percentage of Gestational diabetes among women with a recent live birth.
"DataValue_dia03" indicates the proportion of diabetes-related mortality across the population.
"DataValue_dia04" measures the percentage of diabetic ketoacidosis mortality.
"LocationAbbr" represents the abbreviations of each state, and "YearStart" indicates the year when data collection began. By using Variance Inflation Factor (VIF) analysis, we verified that "DataValue_dia02," "DataValue_dia03," and "DataValue_dia04" each had a VIF value less than 5, suggesting negligible multicollinearity among these variables. This finding justified the inclusion of these three continuous variables, along with "YearStart" and "LocationAbbr," as predictors in our model.
Our analysis found that Linear Regression yielded an impressive R-square of 0.9836 and the lowest MSE of 0.5155. Although the Decision Tree model showed a perfect R-square of 1, this was indicative of overfitting. Random Forest and Ridge Regression also performed well, with R-square values of 0.9227 and 0.9148, and MSEs of 3.2373 and 1.1676, respectively. However, considering the superior interpretability and outstanding performance of the Linear Regression model, we decided it was the most appropriate model for our study. This decision allows us to clearly communicate the impact of various factors on diabetes prevalence, which is essential for informing public health policies and interventions.

In our final linear model, we have found that the p-value for all the dummy variables of LocationAbbr, representing states, are less than 0.05, indicating that they are statistically significant. Therefore, we can see that location, or states, in this case, is most significant in predicting the percentage of Diabetes patients among adults. This is in-line with our Exploratory Data Analysis, as we discovered that the percentage of diabetes patients among adults differ a lot between states. Specifically, deep south states seem to have a higher percentage of diabetes patients among adults than other places in America, and West Virginia has the highest percentage of diabetes patients among all the states in America.
""")
    ]),

])    

# TODO: Separate into modules
# Callback for updating the bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    Input('categorical-dropdown', 'value')
)
def update_bar_chart(selected_column):
    return vp.update_bar_chart(df, selected_column)

# Callback for updating the boxplot
@app.callback(
    Output('boxplot-chart', 'figure'),
    [Input('quantitative-dropdown', 'value'),
     Input('group-dropdown', 'value')]  # Take input from the new group dropdown as well
)
def update_boxplot(selected_quantitative, selected_group):
    return vp.update_boxplot(df, selected_quantitative, selected_group)

# Callback for updating the histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    Input('quantitative-variable-dropdown', 'value')
)
def update_histogram(selected_variable):
    return vp.update_histogram(df, selected_variable)

def create_bar_chart_for_dia(dia_number):
    return vp.create_bar_chart_for_dia(df, dia_number)

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
    return vp.update_pair_plot(final_df)

# Callback for updating the us-heatmap for counts
@app.callback(
    Output('us-heatmap', 'figure'),
    Input('table', 'data')
)
def update_us_heatmap_state(_):
    return vp.update_us_heatmap_state(df, state_count)

# Callback for us-heatmap for DIA01
@app.callback(
    Output('us-heatmap-DIA01', 'figure'),
    [Input('year-slider-1', 'value')]
)
def update_us_heatmap(selected_year, question_id='DIA01'):
    return vp.update_us_heatmap(df, selected_year, question_id)


# Callback for us-heatmap for DIA02
@app.callback(
    Output('us-heatmap-DIA02', 'figure'),
    [Input('year-slider-2', 'value')]
)
def update_us_heatmap(selected_year, question_id='DIA02'):
    return vp.update_us_heatmap(df, selected_year, question_id)

# Callback for state and race
@app.callback(
    Output('pivot_table', 'figure'),
    Input('year-slider-3', 'value')
)
def state_race(selected_year):
    return vp.state_race(df, selected_year)

# Create gender comparison bar plot
@app.callback(
    Output('gender-bar-plot', 'figure'),
    Input('year-slider-4', 'value')
)

def gender_bar_plot(selected_year):
    return vp.gender_bar_plot(df, selected_year)

@app.callback(
    [Output('trend-chart-dia01', 'figure'),
     Output('trend-chart-dia02', 'figure'),
     Output('trend-chart-dia03', 'figure'),
     Output('trend-chart-dia04', 'figure')],
    [Input('state-dropdown', 'value')]
)

def update_trend_charts(selected_state):
    return vp.update_trend_charts(df, selected_state)

# Callback for the correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('table', 'data')]
)
def update_correlation_heatmap(selected_year):
    return vp.update_correlation_heatmap(final_df, selected_year)

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
def update_output(n_clicks, location, year, dia02, dia03, dia04):
    return um.get_prediction_text(n_clicks, location, year, dia02, dia03, dia04)

if __name__ == '__main__':
    app.run_server(debug=True)
