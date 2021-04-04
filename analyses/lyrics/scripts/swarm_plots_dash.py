"""
Produce swarm plot illustrating the vocabularies of bands.
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import lyrics_utils as utils
from basic_plots import get_word_stats


plt.switch_backend('Agg')


def songs2bands(data):
    genre_cols = [c for c in data.columns if 'genre_' in c]
    out = pd.concat(
        (
            data.groupby('band_id')[['band_name', 'band_genre']].first(),
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
        'genre',
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


def uniq_first_words(x, num_words):
    return len(set(x[:num_words]))


def get_band_words(data, num_bands=None, num_words=None):
    data_short = data[data['word_count'] > num_words].copy()
    top_reviewed_bands = data_short.sort_values('reviews')['name'][-num_bands:]
    data_short = data_short.loc[top_reviewed_bands.index]
    data_short['unique_first_words'] = data_short['words'].apply(uniq_first_words, args=(num_words,))
    data_short = data_short.sort_values('unique_first_words')
    return data_short


def get_swarm_data(series):
    markersize = 20
    fig = plt.figure(figsize=(25, 10))
    ax = sns.swarmplot(x=series, size=markersize)
    swarm_pts = ax.collections[0].get_offsets()
    swarm_data = pd.DataFrame(
        swarm_pts,
        index=series.index,
        columns=['x', 'y']
    )
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    swarm_props = {
        'xlim': ax.get_xlim(),
        'ylim': ax.get_ylim(),
        'axsize': (bbox.width * fig.dpi, bbox.height * fig.dpi),
        'markersize': markersize
    }
    plt.close()
    return swarm_data, swarm_props


def add_scatter(fig, data, size, opacity=1.0):
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=data['x'],
            y=data['y'],
            customdata=data[['name', 'genre']],
            opacity=opacity,
            marker=dict(
                size=size,
                color='#1f77b4',
                line=dict(width=2, color='DarkSlateGray')
            ),
            hovertemplate='<b>%{customdata[0]}</b><br><br>'
                          'Unique words: %{x:.0f}<br>'
                          'Genre: %{customdata[1]}',
            name='',
        )
    )


def plot_scatter(data, filter_columns, sns_props, plotly_props):
    xlim = sns_props['xlim']
    ylim = sns_props['ylim']
    axsize = sns_props['axsize']
    markersize = sns_props['markersize']

    fig = go.Figure()
    if len(filter_columns) > 0:
        filt = (data[filter_columns] > 0).any(axis=1)
        bright = data[filt]
        dim = data[~filt]
        add_scatter(fig, bright, markersize)
        add_scatter(fig, dim, markersize, opacity=0.1)
    else:
        add_scatter(fig, data, markersize)

    fig.update_layout(
        autosize=False,
        width=axsize[0],
        height=axsize[1],
        showlegend=False,
        hoverlabel=dict(bgcolor='#730000', font_color='#EBEBEB'),
        template='plotly_dark',
    )
    fig.update_xaxes(range=xlim)
    fig.update_yaxes(range=ylim)
    return fig


app = dash.Dash(__name__)

cfg = utils.get_config()
output = cfg['output']
if not os.path.exists(output):
    os.mkdir(output)
song_df = utils.load_songs(cfg['input'])
song_df = get_word_stats(song_df)
band_df = songs2bands(song_df)
genres = [c for c in band_df.columns if 'genre_' in c]

attrs = {
    'unique_first_words': f"Number of unique words in first {cfg['num_words']} words",
    'word_count': 'Total number of words in discography',
    'words_per_song': 'Average words per song',
    'unique_words_per_song': 'Average unique words per song',
    'seconds_per_song': 'Average song length in seconds',
    'words_per_second': 'Average words per second',
    'unique_words_per_second': f"Average unique words per second",
}

dropdown_attr = dcc.Dropdown(
    id="dropdown_attr",
    options=[
        {'label': v, 'value': k}
        for k, v in attrs.items()
    ],
    value=list(attrs.keys())[0],
    clearable=False,
    style={'background-color': '#111111', 'verticalAlign': 'middle'}
)

dropdown_genre = dcc.Dropdown(
    id="dropdown_genre",
    options=[
        {'label': g.replace('genre_', ''), 'value': g}
        for g in genres
    ],
    clearable=False,
    multi=True,
    style={'background-color': '#111111', 'verticalAlign': 'middle'}
)

app.layout = html.Div([
    html.Div(
        [
            html.H1(f"{cfg['num_bands']} most-reviewed artists"),
            html.Div(
                [
                    html.P(
                        "Attribute:",
                        style={'margin-right': '2em', 'width': '20%'}
                    ),
                    html.Div(
                        [dropdown_attr],
                        className='dropdown',
                        style={'color': 'blue', 'width': '80%'}
                    )
                ],
                style={'width': '500px', 'display': 'flex'}
            ),
            html.Div(
                [
                    html.P(
                        "Filter by genre:",
                        style={'margin-right': '2em', 'width': '30%'}
                    ),
                    html.Div(
                        [dropdown_genre],
                        className='dropdown',
                        style={'width': '70%'}
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
    [Input("dropdown_attr", "value"), Input("dropdown_genre", "value")])
def display_plot(attr, cols):
    band_words = get_band_words(band_df, num_bands=cfg['num_bands'], num_words=cfg['num_words'])
    swarm, swarm_props = get_swarm_data(band_words[attr])
    swarm_df = pd.concat((band_words, swarm), axis=1)
    if cols is None:
        cols = []
    fig = plot_scatter(swarm_df, cols, swarm_props, dict(hover_name='name', orientation='h'))
    return fig


app.run_server(debug=True)
