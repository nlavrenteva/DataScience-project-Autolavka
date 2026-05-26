import plotly.express as px

if 'route_id' not in data.columns:
    data = data.merge(to_route[['village', 'lat', 'lon', 'route_id']],
                      on=['village', 'lat', 'lon'], how='left')
    data['route_id'] = data['route_id'].fillna(-1).astype(int)

print(f'Деревень в зонах обслуживания: {(data["route_id"] >= 0).sum()} из {len(data)}')

preview = px.scatter_mapbox(
    data[data['route_id'] >= 0], lat='lat', lon='lon', color='route_id',
    hover_name='village', hover_data={'population': True, 'predicted_revenue': ':,'},
    zoom=5, height=500, color_continuous_scale='Turbo')
preview.update_layout(mapbox_style='open-street-map',
                      margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
                      title='Деревни, раскрашенные по зоне обслуживания')
preview.show()

from dash import Dash, dcc, html, Input, Output, dash_table

app = Dash(__name__)
app.title = 'Автолавка Арбузерс'
max_rev = int(data['predicted_revenue'].max())

app.layout = html.Div([
    html.H1('Автолавка Арбузерс — деревни и маршруты',
            style={'textAlign': 'center', 'color': '#2C5F2D'}),
    html.Div([
        html.Label('Минимальная прогнозируемая выручка:'),
        dcc.Slider(id='min-revenue', min=0, max=max_rev, step=1000, value=5000,
                   marks={i: f'{i//1000}k' for i in range(0, max_rev + 1, 20000)}),
        dcc.RadioItems(id='filter-routed',
                       options=[{'label': 'Все деревни', 'value': 'all'},
                                {'label': 'Только в маршрутах', 'value': 'routed'}],
                       value='routed', inline=True),
    ], style={'padding': '10px'}),
    dcc.Graph(id='map-graph'),
    html.Div([
        dcc.Graph(id='revenue-hist', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='route-bar', style={'width': '50%', 'display': 'inline-block'}),
    ]),
    html.H3('Топ-20 деревень'),
    dash_table.DataTable(
        id='top-table',
        columns=[{'name': c, 'id': c} for c in
                 ['village', 'population', 'avg_income', 'predicted_revenue', 'route_id']],
        page_size=20, style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': '#2C5F2D', 'color': 'white', 'fontWeight': 'bold'}),
])


@app.callback(
    [Output('map-graph', 'figure'), Output('revenue-hist', 'figure'),
     Output('route-bar', 'figure'), Output('top-table', 'data')],
    [Input('min-revenue', 'value'), Input('filter-routed', 'value')])
def update(min_rev, filt):
    d = data[data['predicted_revenue'] >= min_rev]
    if filt == 'routed':
        d = d[d['route_id'] >= 0]

    fig_map = px.scatter_mapbox(
        d, lat='lat', lon='lon',
        color='route_id' if filt == 'routed' else 'predicted_revenue',
        hover_name='village',
        hover_data={'population': True, 'avg_income': True, 'predicted_revenue': ':,'},
        zoom=5, height=520)
    fig_map.update_layout(mapbox_style='open-street-map',
                          margin={'l': 0, 'r': 0, 't': 30, 'b': 0})

    fig_hist = px.histogram(d, x='predicted_revenue', nbins=40,
                            title='Распределение прогноза выручки',
                            color_discrete_sequence=['#2C5F2D'])

    if filt == 'routed' and not d.empty:
        agg = d.groupby('route_id').agg(total=('predicted_revenue', 'sum'),
                                        n=('village', 'count')).reset_index()
        fig_bar = px.bar(agg, x='route_id', y='total', text='n',
                         title='Суммарная выручка по зоне',
                         color='total', color_continuous_scale='Greens')
        fig_bar.update_traces(texttemplate='%{text} дер.', textposition='outside')
    else:
        fig_bar = px.bar(title='Включите фильтр «Только в маршрутах»')

    top = d.nlargest(20, 'predicted_revenue')[
        ['village', 'population', 'avg_income', 'predicted_revenue', 'route_id']].to_dict('records')
    return fig_map, fig_hist, fig_bar, top

app.run(debug=False, port=8050, jupyter_mode='inline')
