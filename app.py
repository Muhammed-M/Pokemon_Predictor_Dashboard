import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
#  LOAD MODELS
#  Folder structure expected:
#
#  Dash Project/
#  ├── app.py
#  ├── Pokemon.csv
#  ├── assets/style.css
#  └── models/
#      ├── predictors/
#      │   ├── PowerPredictor.pkl
#      │   ├── LegendaryPredictor.pkl
#      │   ├── GenerationPredictor.pkl
#      │   └── NamePredictor.pkl
#      └── encoders/
#          ├── power_encoder.pkl
#          ├── legendary_encoder.pkl
#          ├── generation_encoder.pkl
#          └── name_encoder.pkl
# ─────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

def load(relative_path):
    return joblib.load(os.path.join(BASE, relative_path))

# Intermediate predictors
model_power      = load("models/predictors/PowerPredictor.pkl")
model_legendary  = load("models/predictors/LegendaryPredictor.pkl")
model_generation = load("models/predictors/GenerationPredictor.pkl")

# Final name predictor
model_name       = load("models/predictors/NamePredictor.pkl")

# Encoders
power_encoder      = load("models/encoders/power_encoder.pkl")
legendary_encoder  = load("models/encoders/legendary_encoder.pkl")
generation_encoder = load("models/encoders/generation_encoder.pkl")
name_encoder       = load("models/encoders/name_encoder.pkl")

# Reference data
data = pd.read_csv(os.path.join(BASE, "Pokemon.csv"))
name_to_id    = dict(zip(data["Name"], data["#"]))
name_to_leg   = dict(zip(data["Name"], data["Legendary"]))
name_to_type1 = dict(zip(data["Name"], data["Type 1"]))
name_to_type2 = dict(zip(data["Name"], data["Type 2"].fillna("")))


# ─────────────────────────────────────────────────────────────
#  PREDICTION PIPELINE
#
#  Step 1: user inputs 6 stats
#  Step 2: 3 intermediate models predict Power, Legendary, Generation
#  Step 3: all 9 features feed into NamePredictor
#  Step 4: inverse transform → Pokémon name + confidence
# ─────────────────────────────────────────────────────────────

BASE_COLS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# Exact column list NamePredictor was trained on.
# Pulled directly from the model so no separate pkl file is needed.
# Columns look like: HP, Attack, ..., Legendary, Generation, Power_Fire, Power_Water, ...
TRAINING_COLUMNS = model_name.feature_names_in_.tolist()

def predict(hp, atk, dfn, spa, spd, spe):
    base = pd.DataFrame(
        [[hp, atk, dfn, spa, spd, spe]],
        columns=BASE_COLS
    )

    # Stage 1 — predict intermediate features from 6 base stats
    power_enc      = model_power.predict(base)[0]
    legendary_enc  = model_legendary.predict(base)[0]
    generation_enc = model_generation.predict(base)[0]

    # Decode to original label space
    power_label      = power_encoder.inverse_transform([power_enc])[0]
    legendary_label  = legendary_encoder.inverse_transform([legendary_enc])[0]
    generation_label = generation_encoder.inverse_transform([generation_enc])[0]

    # Stage 2 — add intermediate predictions, manually set the one-hot Power column
    full = base.copy()
    full["Legendary"]  = legendary_label
    full["Generation"] = generation_label
    # Do NOT add Power as a raw column — set the specific Power_X column to 1
    # then reindex fills everything else with 0, matching training exactly
    full[f"Power_{power_label}"] = 1
    full = full.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    # Stage 3 — predict name + confidence
    name_enc   = model_name.predict(full)[0]
    proba      = model_name.predict_proba(full)[0]
    confidence = round(float(proba[name_enc]) * 100, 1)
    name       = name_encoder.inverse_transform([name_enc])[0]

    return name, confidence, power_label, bool(legendary_label), int(generation_label)


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

TYPE_COLORS = {
    "Bug":      "#729130", "Dark":     "#4F3A2D", "Dragon":   "#4052C8",
    "Electric": "#D4B800", "Fairy":    "#C06FBF", "Fighting": "#C12256",
    "Fire":     "#E8590C", "Flying":   "#6080B8", "Ghost":    "#4A5A9A",
    "Grass":    "#4A9E43", "Ground":   "#C4A030", "Ice":      "#5AAEB8",
    "Normal":   "#787A82", "Poison":   "#8A3AAA", "Psychic":  "#D04050",
    "Rock":     "#A89060", "Steel":    "#356B88", "Water":    "#2E7AB8",
}
TYPE_TEXT = {k: "#fff" for k in TYPE_COLORS}
TYPE_TEXT.update({"Electric": "#1a1200", "Ground": "#1a1200"})

STAT_META = [
    #  stat       question                                       min  max  default  color
    ("HP",       "How would you describe your stamina?",         1,   255, 65,  "#1D9E75"),
    ("Attack",   "How hard do you hit back when challenged?",    5,   190, 79,  "#E8590C"),
    ("Defense",  "How well do you handle taking hits?",          5,   230, 73,  "#2E7AB8"),
    ("Sp. Atk",  "How sharp is your mind under pressure?",      10,  194, 72,  "#D4B800"),
    ("Sp. Def",  "How resilient are you mentally?",             20,  230, 71,  "#4A9E43"),
    ("Speed",    "How fast do you move through life?",           5,   180, 68,  "#C06FBF"),
]

SLIDER_IDS = [
    f"slider-{s.lower().replace(' ', '_').replace('.', '')}"
    for s, *_ in STAT_META
]


# ─────────────────────────────────────────────────────────────
#  COMPONENT BUILDERS
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  CHART GENERATORS (NEW)
# ─────────────────────────────────────────────────────────────
def empty_figure():
    """Returns a clean, empty figure for before the user hits predict."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="Awaiting Prediction...", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#ccc"))]
    )
    return fig

def generate_radar_chart(user_stats, predicted_name):
    pred_data = data[data["Name"] == predicted_name].iloc[0]
    pred_stats = [pred_data[col] for col in BASE_COLS]
    
    categories = BASE_COLS + [BASE_COLS[0]]
    u_stats = user_stats + [user_stats[0]]
    p_stats = pred_stats + [pred_stats[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=u_stats, theta=categories, fill='toself', name='You', line_color='#ef5350'))
    fig.add_trace(go.Scatterpolar(r=p_stats, theta=categories, fill='toself', name=predicted_name, line_color='#2E7AB8'))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 255])),
        showlegend=True, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def empty_twins_panel():
    """Returns placeholder text for the Twins panel before prediction."""
    return html.P("Awaiting Prediction...", className="await-text")

def generate_twins_list(user_stats):
    """Generates a clean HTML list of the Top 10 closest Pokémon."""
    stat_matrix = data[BASE_COLS].values
    distances = np.linalg.norm(stat_matrix - user_stats, axis=1)
    
    # Get top 10 closest
    closest_idx = np.argsort(distances)[:10]
    twins = data.iloc[closest_idx]['Name'].tolist()
    
    items = []
    for i, name in enumerate(twins):
        items.append(html.Div([
            html.Span(f"#{i+1}", className="twin-rank"),
            html.Span(name)
        ], className="twin-item"))
        
    return html.Div(items, className="twins-list")

def generate_scatter_chart(user_stats):
    """Generates a clean scatter plot with grey background dots and a glowing gold user pin."""
    # Plot all Pokemon as subtle grey dots, no color mapping or legend
    fig = px.scatter(data, x="Attack", y="Defense", hover_data=["Name"])
    fig.update_traces(marker=dict(color='#e0e0e0', size=6), opacity=0.7)
    
    # Outer Glow Layer for the User Point
    fig.add_trace(go.Scatter(
        x=[user_stats[1]], y=[user_stats[2]], # 1 is Attack, 2 is Defense
        mode='markers',
        marker=dict(size=40, color='rgba(254, 202, 27, 0.25)'), # Transparent Gold
        showlegend=False, hoverinfo='skip'
    ))
    
    # Inner Solid Gold User Point
    fig.add_trace(go.Scatter(
        x=[user_stats[1]], y=[user_stats[2]],
        mode='markers+text',
        marker=dict(size=18, color='#feca1b', line=dict(width=3, color='#ffffff')),
        # Wrapped the text in <b> tags for bolding
        text=["<b>YOU ARE HERE</b>"], textposition="top center", 
        # Removed the invalid 'weight' property
        textfont=dict(color="#d97706", size=12),
        name="You", showlegend=False
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False # Ensure no legend shows up
    )
    return fig

def get_sprite_url(name):
    pid = name_to_id.get(name)
    if pid:
        return (
            "https://raw.githubusercontent.com/PokeAPI/sprites/master"
            f"/sprites/pokemon/other/official-artwork/{pid}.png"
        )
    return ""


def make_slider(stat, question, min_val, max_val, default, color):
    sid = f"slider-{stat.lower().replace(' ', '_').replace('.', '')}"
    return html.Div([
        html.P(question, className="slider-question"),
        html.Div([
            dcc.Slider(
                id=sid,
                min=min_val, max=max_val, step=1, value=default,
                marks=None,
                tooltip={"placement": "right", "always_visible": True},
                className="pokemon-slider",
                updatemode="drag",
            ),
            html.Span(
                stat, className="stat-label",
                style={"color": color, "borderColor": color}
            ),
        ], className="slider-row"),
    ], className="slider-block")


def empty_right_panel():
    return html.Div([
        html.Div(
            html.Div("?", className="artwork-placeholder"),
            className="artwork-wrap"
        ),
        html.P("Set your stats and press the button", className="await-text"),
    ], className="right-inner")


def result_right_panel(name, confidence, power_label, is_legendary, generation):
    sprite_url = get_sprite_url(name)
    type1      = name_to_type1.get(name, "Normal")
    type2      = name_to_type2.get(name, "")

    badges = [
        html.Span(type1, className="type-badge",
                  style={"background": TYPE_COLORS.get(type1, "#787A82"),
                         "color":      TYPE_TEXT.get(type1, "#fff")})
    ]
    if type2:
        badges.append(
            html.Span(type2, className="type-badge",
                      style={"background": TYPE_COLORS.get(type2, "#787A82"),
                             "color":      TYPE_TEXT.get(type2, "#fff")})
        )

    name_class = "pokemon-name legendary-name" if is_legendary else "pokemon-name"
    pokedex_id = name_to_id.get(name, 0)

    return html.Div([

        # Artwork
        html.Div(
            html.Img(src=sprite_url, className="pokemon-img", alt=name),
            className=f"artwork-wrap {'legendary-glow' if is_legendary else ''}"
        ),

        # Name + optional legendary badge
        html.Div([
            html.H2(name, className=name_class),
            html.Span("✦ Legendary", className="legendary-badge")
            if is_legendary else None,
        ], className="name-wrap"),

        # Type badges
        html.Div(badges, className="badges-row"),

        # Generation + Pokédex number
        html.P(
            f"Generation {generation}  ·  #{pokedex_id:03d}",
            className="gen-label"
        ),

        # Confidence bar
        html.Div([
            html.Div([
                html.Span("Model confidence", className="conf-label"),
                html.Span(f"{confidence}%",  className="conf-value"),
            ], className="conf-header"),
            html.Div(
                html.Div(className="conf-fill",
                         style={"width": f"{confidence}%"}),
                className="conf-track"
            ),
        ], className="conf-block"),

        # Reset
        html.Button(
            "↺  try different stats",
            id="btn-reset",
            className="reset-btn",
            n_clicks=0
        ),

    ], className="right-inner")


# ─────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="Which Pokémon Are You?",
    suppress_callback_exceptions=True   # btn-reset lives inside dynamic content
)
server = app.server  # needed for Render / Railway deployment

app.layout = html.Div([

    html.Header([
        html.H1("Which Pokémon Are You?"),
        html.P("answer honestly — your stats don't lie"),
    ], className="app-header"),

    html.Main([
        # --- ROW 1: Existing Left & Right Panels ---
        html.Div([
            html.Section([
                html.H3("Tell us about yourself", className="panel-title"),
                html.Hr(className="panel-divider"),
                *[make_slider(*m) for m in STAT_META],
                html.Hr(className="panel-divider"),
                html.Button("Find my Pokémon →", id="btn-predict", className="predict-btn", n_clicks=0),
            ], className="left-panel"),

            html.Section(id="right-panel", children=empty_right_panel(), className="right-panel"),
        ], className="main-layout"),

        # --- ROW 2: Stat Twins & Radar Chart ---
        html.Div([
            html.Section([
                html.H3("Your Stat Twins", className="panel-title"),
                # Changed from dcc.Graph to html.Div
                html.Div(id="panel-twins", children=empty_twins_panel())
            ], className="square-panel", style={"overflowY": "auto"}), # Added scrolling just in case

            html.Section([
                html.H3("Stat Comparison", className="panel-title"),
                dcc.Graph(id="chart-radar", figure=empty_figure(), style={"height": "350px"})
            ], className="square-panel"),
        ], className="secondary-layout"),


        # --- ROW 3: Global Feature Space ---
        html.Div([
            html.Section([
                html.H3("Global Feature Space", className="panel-title"),
                dcc.Graph(id="chart-scatter", style={"height": "400px"})
            ], className="wide-panel"),
        ], className="tertiary-layout"),

    ], className="main-layout-wrapper"), # Note: wrap everything in a master div if needed,

], className="app-root")


# ─────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────

@app.callback(
    [Output("right-panel", "children"),
     Output("panel-twins", "children"), # Changed to panel-twins / children
     Output("chart-radar", "figure"),
     Output("chart-scatter", "figure")],
    Input("btn-predict", "n_clicks"),
    [State(sid, "value") for sid in SLIDER_IDS],
    prevent_initial_call=True,
)
def handle_prediction(n_predict, hp, atk, dfn, spa, spd, spe):
    user_stats = [hp, atk, dfn, spa, spd, spe]
    
    name, confidence, power_label, is_legendary, generation = predict(
        hp, atk, dfn, spa, spd, spe
    )
    
    panel_ui = result_right_panel(name, confidence, power_label, is_legendary, generation)
    
    # Generate our new HTML list instead of a chart
    ui_twins = generate_twins_list(user_stats) 
    
    fig_radar = generate_radar_chart(user_stats, name)
    fig_scatter = generate_scatter_chart(user_stats)

    return panel_ui, ui_twins, fig_radar, fig_scatter

if __name__ == "__main__":
    app.run(debug=True)
