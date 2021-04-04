"""
Produce scatter plots of lyrical complexity measures computed
in lyrical_complexity_pre.py
"""

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import lyrics_utils as utils
from swarm_plots_dash import get_genre_label, get_swarm_data, plot_scatter


def get_features(**kwargs):
    all_features = {
        'types': 'Types',
        'TTR': 'Type-to-Token Ratio (TTR)',
        'logTTR': 'Log-corrected TTR',
        'MTLD': 'MTLD',
        'logMTLD': 'Log(MTLD)',
        'vocd-D': 'vocd-D',
        'logvocd-D': 'Log(vocd-D)',
    }
    return {k: v for k, v in all_features.items() if kwargs.get(k, False)}


if __name__ == '__main__':
    cfg = utils.get_config(required=('input',))
    figsize = (cfg['fig_width'], cfg['fig_height'])
    markersize = cfg['markersize']
    df = utils.load_bands(cfg['input'])
    features = get_features(**cfg['measures'])
    genres = [c for c in df.columns if 'genre_' in c]

    app = dash.Dash(__name__)

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
            {'label': get_genre_label(df, g), 'value': g}
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
                html.H1(f"Lyrical Complexity of the Top {cfg['num_bands']} Artists"),
                html.P("This interactive swarm plot shows lexical complexity measures"
                       "for the most-reviewed artists."),
                html.Div(
                    [
                        html.P(
                            "Plot feature",
                            style={'margin-right': '2em'}
                        ),
                        html.Div([dropdown_feature], className='dropdown', style={'width': '60%'})
                    ],
                    style={'display': 'flex'}
                ),
                html.Div(
                    [
                        html.P(
                            "Filter by genre:",
                            style={'margin-right': '2em'}
                        ),
                        html.Div([dropdown_genre], className='dropdown', style={'width': '30%'})
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
        [Input("dropdown_feature", "value"),
         Input("dropdown_genre", "value"),
         Input("radio_genre", "value")])
    def display_plot(feature, cols, selection):
        df_sort = df.sort_values(feature)
        swarm, swarm_props = get_swarm_data(df_sort[feature], figsize=figsize, markersize=markersize)
        swarm_df = pd.concat((df_sort, swarm), axis=1)
        if cols is None:
            cols = []
        fig = plot_scatter(swarm_df, cols, swarm_props, union=(selection == 'union'))
        return fig

    app.run_server(debug=True)
