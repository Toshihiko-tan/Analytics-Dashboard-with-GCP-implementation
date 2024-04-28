# This module contains functions to visualize and predict the data 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def update_bar_chart(df, selected_column):
    if selected_column:
        fig = px.bar(
            df[selected_column].value_counts().reset_index(),
            x='index',
            y=selected_column,
            title=f"Counts of {selected_column}"
        )
        return fig
    return go.Figure()

def update_boxplot(df, selected_quantitative, selected_group):
    if selected_quantitative and selected_group:
        fig = px.box(df, y=selected_quantitative, x=selected_group, 
                     title=f"Distribution of {selected_quantitative} by {selected_group}", notched=True)
        return fig
    return {}

def update_histogram(df, selected_variable):
    # Create the histogram using Plotly Express
    fig = px.histogram(df, x=selected_variable, nbins=30, title=f'Distribution of {selected_variable}')
    return fig

def create_bar_chart_for_dia(df, dia_number):
    # Filter the DataFrame for the specified DIA number
    filtered_df = df[df['QuestionID'] == dia_number]
    # Aggregate by state
    aggregated_data = filtered_df.groupby('LocationAbbr').size().reset_index(name='Count')
    # Create the bar chart
    fig = px.bar(aggregated_data, x='LocationAbbr', y='Count', title=f'Counts for {dia_number}')
    return fig

def update_pair_plot(final_df):
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

def update_us_heatmap_state(df, state_count):
    fig = px.choropleth(
        state_count, 
        locations='LocationAbbr', 
        locationmode="USA-states", 
        color='Count',  # This should be the column from your DataFrame that holds the counts
        scope="usa",
        title="Value Counts by State"
    )
    return fig

def update_us_heatmap(df, selected_year, question_id):
    filtered_df = df[(df['QuestionID'] == question_id) &
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

def state_race(df, selected_year):
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

def gender_bar_plot(df, selected_year):
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

def update_trend_charts(df, selected_state):
    figures = []
    for dia in ['DIA01', 'DIA02', 'DIA03', 'DIA04']:
        # Filter data for the selected state and QuestionID
        filtered_data = df[
            (df['LocationAbbr'] == selected_state) &
            (df['QuestionID'] == dia)
        ]
        # Aggregate counts by year
        yearly_counts = filtered_data.groupby('Year').size().reset_index(name='Mortality Rate Count')
        # Create the line chart
        fig = px.line(
            yearly_counts, 
            y='Mortality Rate Count', 
            x='Year', 
            title=f"{dia} Trend in {selected_state}"
        )

        fig.update_xaxes(dtick=1)
        figures.append(fig)
    return figures

def update_correlation_heatmap(final_df, selected_year):
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
