import os
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import lyrics_utils as utils
from basic_plots import get_word_stats


def songs2bands(data):
    genre_cols = [c for c in data.columns if 'genre_' in c]
    out = pd.concat(
        (
            data.groupby('band_id')['band_name'].first(),
            data.groupby(['band_id', 'album_name'])['album_review_num'].first().groupby('band_id').sum(),
            data.groupby(['band_id', 'album_name'])['album_review_avg'].first().groupby('band_id').mean(),
            data.groupby('band_id')['song_words'].sum(),
            data.groupby('band_id')['word_count'].sum(),
            data.groupby('band_id')[[
                'word_count',
                'word_count_uniq',
                'seconds',
                'word_rate',
                'word_rate_uniq',
            ]].mean(),
            data.groupby('band_id')[genre_cols].first(),
        ),
        axis=1
    )
    out.columns = [
        'name',
        'reviews',
        'rating',
        'words',
        'word_count',
        'words_per_song',
        'unique_words_per_song',
        'seconds_per_song',
        'words_per_second',
        'unique_words_per_second',
    ] + genre_cols
    return out


cfg = utils.get_config()
output = cfg['output']
if not os.path.exists(output):
    os.mkdir(output)
song_df = utils.load_songs(cfg['input'])
song_df = get_word_stats(song_df)
band_df = songs2bands(song_df)
num_bands = cfg['num_bands']
top_reviewed = band_df.sort_values('reviews')['name'][-num_bands:]
band_df = band_df.loc[top_reviewed.index]
genres = [c for c in band_df.columns if 'genre_' in c]


app = dash.Dash(__name__)

variables = {
    'reviews': 'Number of album reviews',
    'rating': 'Average album review score',
    'word_count': 'Total word count',
    'words_per_song': 'Words per song',
    'unique_words_per_song': 'Unique words per song',
    'seconds_per_song': 'Average song duration [s]',
    'words_per_second': 'Words per second',
    'unique_words_per_second': 'Unique words per second',
}

app.layout = html.Div([
    html.H1(f"{num_bands} most-reviewed artists"),
    html.Div(
        [
            html.P(
                "X-axis",
                style={'margin-right': '2em'}
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="dropdown_x",
                        options=[
                            {'label': v, 'value': k}
                            for k, v in variables.items()
                        ],
                        value='words_per_song',
                        clearable=False,
                        style={'width': '60%', 'verticalAlign': 'middle'}
                    )
                ],
                style={'width': '60%'}
            )
        ],
        style={'display': 'flex'}
    ),
    html.Div(
        [
            html.P(
                "Y-axis",
                style={'margin-right': '2em'}
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="dropdown_y",
                        options=[
                            {'label': v, 'value': k}
                            for k, v in variables.items()
                        ],
                        value='unique_words_per_song',
                        clearable=False,
                        style={'width': '60%', 'verticalAlign': 'middle'}
                    )
                ],
                style={'width': '60%'}
            )
        ],
        style={'display': 'flex'}
    ),
    html.Div(
        [
            html.P(
                "Filter by genre:",
                style={'margin-right': '2em'}
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="dropdown_z",
                        options=[
                            {'label': g.replace('genre_', ''), 'value': g}
                            for g in ['Show all'] + genres
                        ],
                        value='Show all',
                        clearable=False,
                        style={'verticalAlign': 'middle'}
                    )
                ],
                style={'width': '30%'}
            )
        ],
        style={'display': 'flex'}
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    [Input("dropdown_x", "value"), Input("dropdown_y", "value"), Input("dropdown_z", "value")])
def display_plot(x, y, z):
    if z != 'Show all':
        band_df_filtered = band_df[band_df[z] > 0]
    else:
        band_df_filtered = band_df
    fig = px.scatter(band_df_filtered, x=x, y=y, labels=variables, hover_name='name')
    fig.update_traces(marker=dict(size=10,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    return fig


app.run_server(debug=True)
