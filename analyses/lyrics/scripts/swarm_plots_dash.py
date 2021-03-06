"""
Produce swarm plot illustrating the vocabularies of bands.
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import lyrics_utils as utils
import nlp


plt.switch_backend('Agg')


def uniq_first_words(x, num_words):
    return len(set(x[:num_words]))


def get_band_words(data, num_bands=None, num_words=None):
    data_short = data[data['word_count'] > num_words].copy()
    top_reviewed_bands = data_short.sort_values('reviews')['name'][-num_bands:]
    data_short = data_short.loc[top_reviewed_bands.index]
    data_short['unique_first_words'] = data_short['words'].apply(uniq_first_words, args=(num_words,))
    data_short = data_short.sort_values('unique_first_words')
    return data_short


def get_genre_label(data, col):
    genre = col.replace('genre_', '')
    genre = genre[0].upper() + genre[1:]
    label = f"{genre} ({data[col].sum()} bands)"
    return label


def get_swarm_data(series, figsize, markersize):
    fig = plt.figure(figsize=figsize)
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
    data['genre_split'] = data['genre'].str.wrap(25).apply(lambda x: x.replace('\n', '<br>'))
    if data['x'].max() - data['x'].min() > 10:
        hovertemplate = '<b>%{customdata[0]}</b><br><br>'\
                        'Value: %{x:.0f}<br>'\
                        'Genre: %{customdata[1]}'
    else:
        hovertemplate = '<b>%{customdata[0]}</b><br><br>'\
                        'Value: %{x:.3g}<br>'\
                        'Genre: %{customdata[1]}'
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=data['x'],
            y=data['y'],
            customdata=data[['name', 'genre_split']],
            opacity=opacity,
            marker=dict(
                size=size,
                color='#1f77b4',
                line=dict(width=2, color='DarkSlateGray')
            ),
            hovertemplate=hovertemplate,
            name='',
        )
    )


def plot_scatter(data, filter_columns, sns_props, union=True):
    xlim = sns_props['xlim']
    ylim = sns_props['ylim']
    axsize = sns_props['axsize']
    size = sns_props['markersize']

    fig = go.Figure()
    if len(filter_columns) > 0:
        if union:
            filt = (data[filter_columns] > 0).any(axis=1)
        else:
            filt = (data[filter_columns] > 0).all(axis=1)
        bright = data[filt]
        dim = data[~filt]
        add_scatter(fig, bright, size)
        add_scatter(fig, dim, size, opacity=0.15)
    else:
        add_scatter(fig, data, size)

    fig.update_layout(
        autosize=False,
        width=axsize[0],
        height=axsize[1],
        showlegend=False,
        hoverlabel=dict(bgcolor='#730000', font_color='#EBEBEB', font_family='Monospace'),
        template='plotly_dark',
    )
    fig.update_xaxes(range=xlim, gridwidth=2, gridcolor='#444444', side='top')
    fig.update_yaxes(range=ylim, gridwidth=2, gridcolor='#444444', tickvals=[0], ticktext=[''])
    return fig


if __name__ == '__main__':
    app = dash.Dash(__name__)

    cfg = utils.get_config(required=('input',))
    figsize = (cfg['fig_width'], cfg['fig_height'])
    markersize = cfg['markersize']
    band_df = utils.load_bands(cfg['input'])
    band_df = nlp.get_band_stats(band_df)
    genres = [c for c in band_df.columns if 'genre_' in c]
    band_words = get_band_words(band_df, num_bands=cfg['num_bands'], num_words=cfg['num_words'])

    features = {
        'unique_first_words': f"Number of unique words in first {cfg['num_words']:,.0f} words",
        'word_count': 'Total number of words in discography',
        'words_per_song': 'Average words per song',
        'words_per_song_uniq': 'Average unique words per song',
        'seconds_per_song': 'Average song length in seconds',
        'word_rate': 'Average words per second',
        'word_rate_uniq': f"Average unique words per second",
    }

    dropdown_feature = dcc.Dropdown(
        id="dropdown_feature",
        options=[
            {'label': v, 'value': k}
            for k, v in features.items()
        ],
        value=list(features.keys())[0],
        clearable=False,
        style={'background-color': '#111111', 'verticalAlign': 'middle'}
    )


    dropdown_genre = dcc.Dropdown(
        id="dropdown_genre",
        options=[
            {'label': get_genre_label(band_words, g), 'value': g}
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
                html.P(f"This interactive swarm plot shows the most-reviewed artists who have at least \
    {cfg['num_words']:,.0f} words in their collection of song lyrics."),
                html.Div(
                    [
                        html.P(
                            "Plot feature:",
                            style={'margin-right': '2em', 'width': '20%'}
                        ),
                        html.Div(
                            [dropdown_feature],
                            className='dropdown',
                            style={'width': '80%'}
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
        [Input("dropdown_feature", "value"), Input("dropdown_genre", "value"), Input("radio_genre", "value")])
    def display_plot(feature, cols, selection):
        band_words_sorted = band_words.sort_values(feature)
        swarm, swarm_props = get_swarm_data(band_words_sorted[feature], figsize=figsize, markersize=markersize)
        swarm_df = pd.concat((band_words_sorted, swarm), axis=1)
        if cols is None:
            cols = []
        fig = plot_scatter(swarm_df, cols, swarm_props, union=(selection == 'union'))
        return fig

    app.run_server(debug=True)
