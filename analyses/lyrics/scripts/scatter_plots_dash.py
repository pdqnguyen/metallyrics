import os
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import lyrics_utils as utils
import nlp


def get_genre_label(data, col):
    genre = col.replace('genre_', '')
    genre = genre[0].upper() + genre[1:]
    label = f"{genre} ({data[col].sum()} bands)"
    return label


def add_scatter(fig, data, x, y, opacity=1.0):
    if data[x].max() - data[x].min() > 10:
        x_template = '%{x:.0f}'
    else:
        x_template = '%{x:.3g}'
    if data[y].max() - data[y].min() > 10:
        y_template = '%{y:.0f}'
    else:
        y_template = '%{y:.3g}'
    hovertemplate = '<b>%{customdata[0]}</b><br><br>'\
                    'X: ' + x_template + '<br>'\
                    'Y: ' + y_template + '<br>'\
                    'Genre: %{customdata[1]}'
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=data[x],
            y=data[y],
            customdata=data[['name', 'genre']],
            opacity=opacity,
            marker=dict(
                size=10,
                color='#1f77b4',
                line=dict(width=2, color='DarkSlateGray')
            ),
            hovertemplate=hovertemplate,
            name='',
        )
    )


def plot_scatter(data, x, y, filter_columns, union=True):
    fig = go.Figure()
    if len(filter_columns) > 0:
        if union:
            filt = (data[filter_columns] > 0).any(axis=1)
        else:
            filt = (data[filter_columns] > 0).all(axis=1)
        bright = data[filt]
        dim = data[~filt]
        add_scatter(fig, bright, x, y)
        add_scatter(fig, dim, x, y, opacity=0.15)
    else:
        add_scatter(fig, data, x, y)
    fig.update_layout(
        width=1200,
        height=800,
        showlegend=False,
        hoverlabel=dict(bgcolor='#730000', font_color='#EBEBEB', font_family='Monospace'),
        template='plotly_dark',
        xaxis_title = features[x],
        yaxis_title = features[y],
    )
    fig.update_xaxes(gridwidth=2, gridcolor='#444444')
    fig.update_yaxes(gridwidth=2, gridcolor='#444444')
    return fig


cfg = utils.get_config()
output = cfg['output']
if not os.path.exists(output):
    os.mkdir(output)
band_df = utils.load_bands(cfg['input'])
band_df = nlp.get_band_stats(band_df)
num_bands = cfg['num_bands']
top_reviewed = band_df.sort_values('reviews')['name'][-num_bands:]
band_df = band_df.loc[top_reviewed.index]
genres = [c for c in band_df.columns if 'genre_' in c]


app = dash.Dash(__name__)

features = {
    'reviews': 'Number of album reviews',
    'rating': 'Average album review score',
    'word_count': 'Total word count',
    'words_per_song': 'Words per song',
    'words_per_song_uniq': 'Unique words per song',
    'seconds_per_song': 'Average song duration [s]',
    'word_rate': 'Words per second',
    'word_rate_uniq': 'Unique words per second',
}

dropdown_x = dcc.Dropdown(
    id="dropdown_x",
    options=[
        {'label': v, 'value': k}
        for k, v in features.items()
    ],
    value='words_per_song',
    clearable=False,
    style={'background-color': '#111111', 'verticalAlign': 'middle'}
)

dropdown_y = dcc.Dropdown(
    id="dropdown_y",
    options=[
        {'label': v, 'value': k}
        for k, v in features.items()
    ],
    value='words_per_song_uniq',
    clearable=False,
    style={'background-color': '#111111', 'verticalAlign': 'middle'}
)

dropdown_z = dcc.Dropdown(
    id="dropdown_z",
    options=[
        {'label': get_genre_label(band_df, g), 'value': g}
        for g in genres
    ],
    clearable=False,
    multi=True,
    style={'background-color': '#111111', 'verticalAlign': 'middle'}
)

radio_genre = dcc.RadioItems(
    id="radio_genre",
    options=[
        {'label': 'Match ANY selected genre', 'value': 'union'},
        {'label': 'Match ALL selected genres', 'value': 'inter'},
    ],
    value='union',
    labelStyle={'display': 'inline-block'}
)

app.layout = html.Div([
    html.Div(
        [
            html.H1(f"Lyrical Properties of the Top {cfg['num_bands']} Artists"),
            html.P(f"This interactive scatter plot shows the most-reviewed artists who have at least \
{cfg['num_words']:,.0f} words in their collection of song lyrics."),
            html.Div(
                [
                    html.P(
                        "X-axis",
                        style={'margin-right': '2em'}
                    ),
                    html.Div([dropdown_x], className='dropdown', style={'width': '60%'})
                ],
                style={'display': 'flex'}
            ),
            html.Div(
                [
                    html.P(
                        "Y-axis",
                        style={'margin-right': '2em'}
                    ),
                    html.Div([dropdown_y], className='dropdown', style={'width': '60%'})
                ],
                style={'display': 'flex'}
            ),
            html.Div(
                [
                    html.P(
                        "Filter by genre:",
                        style={'margin-right': '2em'}
                    ),
                    html.Div([dropdown_z], className='dropdown', style={'width': '30%'})
                ],
                style={'display': 'flex'}
            ),
            html.Div(
                [
                    html.P(
                        "Filter mode:",
                        style={'margin-right': '2em', 'width': '20%'}
                    ),
                    html.Div(
                        [radio_genre],
                        className='radio',
                        style={'width': '80%'}
                    )
                ],
                style={'width': '500px', 'display': 'flex'}
            ),
            dcc.Graph(id="graph"),
        ],
        style={
            'width': '800px',
            'font-family': 'Helvetica',
        }
    )
], style={})


@app.callback(
    Output("graph", "figure"),
    [Input("dropdown_x", "value"),
     Input("dropdown_y", "value"),
     Input("dropdown_z", "value"),
     Input("radio_genre", "value")])
def display_plot(x, y, z, selection):
    if z is None:
        z = []
    fig = plot_scatter(band_df, x, y, z, union=(selection == 'union'))
    return fig


app.run_server(debug=True)
