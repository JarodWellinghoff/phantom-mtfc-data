import dash
import os
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self):
        frames = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".csv"):
                frame = {}
                filepath = os.path.join(self.directory, filename)
                technique, vmi, insert = filename.replace(".csv", "").split("_")
                df = pd.read_csv(filepath, header=None)
                spatial_location = self.get_values(df, "Spatial Location (cm)")
                line_spread_function = self.get_values(df, "Line Spread Function (LSF)")
                edge_spread_function = self.get_values(df, "Edge Spread Function (ESF)")
                spatial_frequency = self.get_values(df, "Spatial Frequency (1/cm)")
                contrast_upper = self.get_values(
                    df, "Contrast Dependent Spatial Resolution (MTFc) upper"
                )
                contrast = self.get_values(
                    df, "Contrast Dependent Spatial Resolution (MTFc)"
                )
                contrast_lower = self.get_values(
                    df, "Contrast Dependent Spatial Resolution (MTFc) lower"
                )
                if spatial_location.size == 0:
                    continue
                if line_spread_function.size == 0:
                    continue
                if edge_spread_function.size == 0:
                    continue
                if spatial_frequency.size == 0:
                    continue
                if contrast_upper.size == 0:
                    continue
                if contrast.size == 0:
                    continue
                if contrast_lower.size == 0:
                    continue

                if insert == "lc":
                    insert = "Low Contrast"
                elif insert == "poly":
                    insert = "Polyethylene"
                if vmi == "t3d":
                    vmi = "T3D"
                else:
                    vmi += " keV"
                # If found, add to the list
                frame["technique"] = technique.upper()
                frame["vmi"] = vmi
                frame["insert"] = insert.title()
                frame["spatial_location"] = spatial_location
                frame["line_spread_function"] = line_spread_function
                frame["edge_spread_function"] = edge_spread_function
                frame["spatial_frequency"] = spatial_frequency
                frame["contrast_upper"] = contrast_upper
                frame["contrast"] = contrast
                frame["contrast_lower"] = contrast_lower
                frame["filename"] = filename
                frames.append(frame)
        return frames

    def get_values(self, df, cell_str):
        for col in range(len(df.columns)):
            row = df[col].eq(cell_str).idxmax()
            if row == 0:
                continue
            values = np.array(
                df.iloc[row + 1 :, col].reset_index(drop=True), dtype=float
            )
            return values
        return np.array([])


# Sample data - replace with your actual frames
frames = [
    {
        "technique": "QIR",
        "vmi": "50 keV",
        "insert": "Acrylic",
        "spatial_location": np.array([0, 1, 2, 3, 4]),
        "line_spread_function": np.array([0.8, 0.9, 1.0, 0.9, 0.8]),
        "edge_spread_function": np.array([0.1, 0.3, 0.7, 0.9, 1.0]),
        "spatial_frequency": np.array([0, 0.5, 1.0, 1.5, 2.0]),
        "contrast": np.array([1.0, 0.8, 0.6, 0.4, 0.2]),
        "contrast_upper": np.array([1.1, 0.9, 0.7, 0.5, 0.3]),
        "contrast_lower": np.array([0.9, 0.7, 0.5, 0.3, 0.1]),
        "filename": "qir_50_acrylic.csv",
    },
    {
        "technique": "FBP",
        "vmi": "55 keV",
        "insert": "Bone",
        "spatial_location": np.array([0, 1, 2, 3, 4]),
        "line_spread_function": np.array([0.7, 0.8, 0.9, 0.8, 0.7]),
        "edge_spread_function": np.array([0.1, 0.4, 0.8, 0.9, 1.0]),
        "spatial_frequency": np.array([0, 0.5, 1.0, 1.5, 2.0]),
        "contrast": np.array([0.9, 0.7, 0.5, 0.3, 0.1]),
        "contrast_upper": np.array([1.0, 0.8, 0.6, 0.4, 0.2]),
        "contrast_lower": np.array([0.8, 0.6, 0.4, 0.2, 0.05]),
        "filename": "fbp_55_bone.csv",
    },
]

# Initialize Dash app
app = dash.Dash(__name__)

# Define available options
TECHNIQUES = ["QIR", "FBP"]
VMIS = ["50 keV", "55 keV", "60 keV", "65 keV", "70 keV", "T3D"]
INSERTS = ["Acrylic", "Air", "Bone", "Low Contrast", "Polyethylene"]


# Add callbacks for dynamic layout styling
@callback(
    [
        Output("whole-container", "style"),
        Output("main-container", "style"),
        Output("title", "style"),
        Output("controls-container", "style"),
        Output("options-container", "style"),
        Output("technique-label", "style"),
        Output("vmi-label", "style"),
        Output("insert-label", "style"),
        Output("plots-to-display-label", "style"),
        Output("options-label", "style"),
        Output("plot-selection", "style"),
        Output("options-toggle", "style"),
    ],
    [Input("options-toggle", "value")],
)
def update_layout_theme(options):
    """Update layout styling based on dark mode"""
    dark_mode = "dark_mode" in (options or [])
    theme = get_theme_styles(dark_mode)

    whole_style = {
        "backgroundColor": theme["bg_color"],
        "width": "100%",
        "margin": "-8px",
    }

    main_style = {
        "maxWidth": "1600px",
        "margin": "0 auto",
        "padding": "30px",
        "backgroundColor": theme["bg_color"],
        "minHeight": "100vh",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        # "transition": "all 0.3s ease",
    }

    title_style = {
        "textAlign": "center",
        "marginBottom": 40,
        "color": theme["title_color"],
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "fontSize": "2.5rem",
        "fontWeight": "300",
        "textShadow": "2px 2px 4px rgba(0,0,0,0.1)",
        # "transition": "color 0.3s ease",
    }

    card_style = {
        "marginBottom": 30,
        "padding": 25,
        "backgroundColor": theme["card_bg"],
        "borderRadius": "15px",
        "boxShadow": (
            "0 8px 25px rgba(0, 0, 0, 0.1)"
            if not dark_mode
            else "0 8px 25px rgba(0, 0, 0, 0.3)"
        ),
        "border": f"1px solid {theme['border_color']}",
        # "transition": "all 0.3s ease",
    }

    technique_label = {
        "fontWeight": "600",
        "color": theme["title_color"],
        "marginBottom": "8px",
        "display": "block",
    }

    plots_to_display_label = {
        "fontWeight": "600",
        "color": theme["title_color"],
        "marginRight": 20,
        "fontSize": "16px",
    }

    plot_selection_style = {
        "display": "inline-block",
        "color": theme["title_color"],
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "fontSize": "14px",
    }

    return (
        whole_style,
        main_style,
        title_style,
        card_style,
        card_style,
        technique_label,
        technique_label,
        technique_label,
        plots_to_display_label,
        plots_to_display_label,
        plot_selection_style,
        plot_selection_style,
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Interactive Data Visualization Dashboard", id="title"),
                # Controls
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Technique:",
                                    id="technique-label",
                                ),
                                dcc.Dropdown(
                                    id="technique-dropdown",
                                    options=[
                                        {"label": t, "value": t} for t in TECHNIQUES
                                    ],
                                    value=TECHNIQUES,
                                    multi=True,
                                    style={"fontFamily": "Segoe UI, Arial, sans-serif"},
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "marginRight": "3%",
                                "verticalAlign": "top",
                            },
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "VMI:",
                                    id="vmi-label",
                                ),
                                dcc.Dropdown(
                                    id="vmi-dropdown",
                                    options=[{"label": v, "value": v} for v in VMIS],
                                    value=VMIS,
                                    multi=True,
                                    style={"fontFamily": "Segoe UI, Arial, sans-serif"},
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "marginRight": "3%",
                                "verticalAlign": "top",
                            },
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Insert:",
                                    id="insert-label",
                                ),
                                dcc.Dropdown(
                                    id="insert-dropdown",
                                    options=[{"label": i, "value": i} for i in INSERTS],
                                    value=INSERTS,
                                    multi=True,
                                    style={"fontFamily": "Segoe UI, Arial, sans-serif"},
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        ),
                    ],
                    id="controls-container",
                ),
                # Plot selection checkboxes
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Plots to Display:",
                                    id="plots-to-display-label",
                                ),
                                dcc.Checklist(
                                    id="plot-selection",
                                    options=[
                                        {
                                            "label": " Line Spread Function",
                                            "value": "lsf",
                                        },
                                        {
                                            "label": " Edge Spread Function",
                                            "value": "esf",
                                        },
                                        {
                                            "label": " Contrast Resolution",
                                            "value": "contrast",
                                        },
                                    ],
                                    value=["lsf", "esf", "contrast"],
                                    inline=True,
                                    style={
                                        "display": "inline-block",
                                        "fontFamily": "Segoe UI, Arial, sans-serif",
                                        "fontSize": "14px",
                                    },
                                ),
                            ],
                            style={"marginBottom": 15},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Options:",
                                    id="options-label",
                                ),
                                dcc.Checklist(
                                    id="options-toggle",
                                    options=[
                                        {
                                            "label": " Show Error Bands",
                                            "value": "show_bands",
                                        },
                                        {"label": " Smooth LSF", "value": "smooth_lsf"},
                                        {"label": " Dark Mode", "value": "dark_mode"},
                                    ],
                                    value=["show_bands"],
                                    inline=True,
                                ),
                            ]
                        ),
                    ],
                    id="options-container",
                ),
                # Plots
                html.Div(id="plots-container"),
            ],
            id="main-container",
        )
    ],
    id="whole-container",
)


def filter_frames(techniques, vmis, inserts):
    """Filter frames based on selected criteria"""
    if not techniques or not vmis or not inserts:
        return []

    return [
        frame
        for frame in frames
        if frame["technique"] in techniques
        and frame["vmi"] in vmis
        and frame["insert"] in inserts
    ]


def generate_title(plot_type, techniques, vmis, inserts):
    """Generate dynamic title based on selected filters"""
    title = plot_type
    if techniques and len(techniques) < len(TECHNIQUES):
        title += f" - {', '.join(techniques)}"
    if vmis and len(vmis) < len(VMIS):
        title += f" - {', '.join(vmis)}"
    if inserts and len(inserts) < len(INSERTS):
        title += f" - {', '.join(inserts)}"
    return title


def smooth_data(y_data, window_size=5):
    """Apply simple moving average smoothing"""
    if len(y_data) < window_size:
        return y_data
    kernel = np.ones(window_size) / window_size
    return np.convolve(y_data, kernel, mode="same")


def get_theme_styles(dark_mode=False):
    """Get theme-specific styles"""
    if dark_mode:
        return {
            "bg_color": "#1e1e1e",
            "card_bg": "#2d2d2d",
            "text_color": "#ffffff",
            "border_color": "#404040",
            "plot_bg": "#2d2d2d",
            "grid_color": "#404040",
            "title_color": "#ffffff",
        }
    else:
        return {
            "bg_color": "#f8fafc",
            "card_bg": "white",
            "text_color": "#2c3e50",
            "border_color": "#e8e9ea",
            "plot_bg": "white",
            "grid_color": "#ecf0f1",
            "title_color": "#2c3e50",
        }


def create_lsf_plot(
    filtered_frames, techniques, vmis, inserts, smooth=False, dark_mode=False
):
    """Create Line Spread Function plot with optional smoothing"""
    fig = go.Figure()
    theme = get_theme_styles(dark_mode)

    # Modern vibrant color palette
    colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#fd79a8",
        "#00b894",
    ]

    for i, frame in enumerate(filtered_frames):
        y_data = frame["line_spread_function"]
        if smooth:
            y_data = smooth_data(y_data)

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=frame["spatial_location"],
                y=y_data,
                mode="lines",
                name=f"{frame['technique']} {frame['vmi']} {frame['insert']}",
                line=dict(color=color, width=4),
                marker=dict(
                    color=color,
                    size=8,
                    symbol="circle",
                    line=dict(width=2, color=theme["plot_bg"]),
                ),
            )
        )

    title = "Line Spread Function"
    if smooth:
        title += " (Smoothed)"

    fig.update_layout(
        title={
            "text": generate_title(title, techniques, vmis, inserts),
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 24,
                "family": "Segoe UI, Arial, sans-serif",
                "color": theme["title_color"],
                "weight": "bold",
            },
        },
        xaxis_title="Spatial Location (cm)",
        yaxis_title="Line Spread Function (LSF)",
        height=550,
        plot_bgcolor=theme["plot_bg"],
        paper_bgcolor=theme["plot_bg"],
        font={
            "family": "Segoe UI, Arial, sans-serif",
            "color": theme["text_color"],
            "size": 14,
        },
        xaxis=dict(
            gridcolor=theme["grid_color"],
            gridwidth=2,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color=theme["text_color"]),
            title_font=dict(size=16, weight="bold", color=theme["text_color"]),
        ),
        yaxis=dict(
            gridcolor=theme["grid_color"],
            gridwidth=2,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color=theme["text_color"]),
            title_font=dict(size=16, weight="bold", color=theme["text_color"]),
        ),
        legend=dict(
            bgcolor=f"rgba({255 if not dark_mode else 45},{255 if not dark_mode else 45},{255 if not dark_mode else 45},0.9)",
            bordercolor="#bdc3c7" if not dark_mode else "#555555",
            borderwidth=2,
            font=dict(size=12, color=theme["text_color"]),
        ),
        margin=dict(l=80, r=80, t=100, b=80),
    )

    return fig


def create_esf_plot(filtered_frames, techniques, vmis, inserts, dark_mode=False):
    """Create Edge Spread Function plot"""
    fig = go.Figure()
    theme = get_theme_styles(dark_mode)

    # Modern vibrant color palette
    colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#fd79a8",
        "#00b894",
    ]

    for i, frame in enumerate(filtered_frames):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=frame["spatial_location"],
                y=frame["edge_spread_function"],
                mode="lines",
                name=f"{frame['technique']} {frame['vmi']} {frame['insert']}",
                line=dict(color=color, width=4),
                marker=dict(
                    color=color,
                    size=8,
                    symbol="circle",
                    line=dict(width=2, color=theme["plot_bg"]),
                ),
            )
        )

    fig.update_layout(
        title={
            "text": generate_title("Edge Spread Function", techniques, vmis, inserts),
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 24,
                "family": "Segoe UI, Arial, sans-serif",
                "color": theme["title_color"],
                "weight": "bold",
            },
        },
        xaxis_title="Spatial Location (cm)",
        yaxis_title="Edge Spread Function (ESF)",
        height=550,
        plot_bgcolor=theme["plot_bg"],
        paper_bgcolor=theme["plot_bg"],
        font={
            "family": "Segoe UI, Arial, sans-serif",
            "color": theme["text_color"],
            "size": 14,
        },
        xaxis=dict(
            gridcolor=theme["grid_color"],
            gridwidth=2,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color=theme["text_color"]),
            title_font=dict(size=16, weight="bold", color=theme["text_color"]),
        ),
        yaxis=dict(
            gridcolor=theme["grid_color"],
            gridwidth=2,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color=theme["text_color"]),
            title_font=dict(size=16, weight="bold", color=theme["text_color"]),
        ),
        legend=dict(
            bgcolor=f"rgba({255 if not dark_mode else 45},{255 if not dark_mode else 45},{255 if not dark_mode else 45},0.9)",
            bordercolor="#bdc3c7" if not dark_mode else "#555555",
            borderwidth=2,
            font=dict(size=12, color=theme["text_color"]),
        ),
        margin=dict(l=80, r=80, t=100, b=80),
    )

    return fig


def create_contrast_plot(
    filtered_frames, techniques, vmis, inserts, show_bands=True, dark_mode=False
):
    """Create Contrast Resolution plot with optional filled error bands"""
    fig = go.Figure()
    theme = get_theme_styles(dark_mode)

    # Modern vibrant color palette
    colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#fd79a8",
        "#00b894",
    ]

    for i, frame in enumerate(filtered_frames):
        name = f"{frame['technique']} {frame['vmi']} {frame['insert']}"
        legend_group = f"group{i}"
        color = colors[i % len(colors)]

        if show_bands:
            # Convert hex to rgba for transparency
            rgb_vals = [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
            rgba_color = f"rgba({rgb_vals[0]}, {rgb_vals[1]}, {rgb_vals[2]}, 0.15)"

            # Add upper bound (invisible line)
            fig.add_trace(
                go.Scatter(
                    x=frame["spatial_frequency"],
                    y=frame["contrast_upper"],
                    mode="lines",
                    line=dict(width=0, color=color),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=legend_group,
                    name=f"{name} upper",
                )
            )

            # Add lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=frame["spatial_frequency"],
                    y=frame["contrast_lower"],
                    mode="lines",
                    line=dict(width=0, color=color),
                    fill="tonexty",
                    fillcolor=rgba_color,
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=legend_group,
                    name=f"{name} band",
                )
            )

        # Add main line
        fig.add_trace(
            go.Scatter(
                x=frame["spatial_frequency"],
                y=frame["contrast"],
                mode="lines",
                name=name,
                legendgroup=legend_group,
                line=dict(color=color, width=4),
                marker=dict(
                    color=color,
                    size=8,
                    symbol="circle",
                    line=dict(width=2, color=theme["plot_bg"]),
                ),
            )
        )

    fig.update_layout(
        title={
            "text": generate_title(
                "Contrast Dependent Spatial Resolution", techniques, vmis, inserts
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 24,
                "family": "Segoe UI, Arial, sans-serif",
                "color": theme["title_color"],
                "weight": "bold",
            },
        },
        xaxis_title="Spatial Frequency (1/cm)",
        yaxis_title="Contrast Dependent Spatial Resolution (MTFc)",
        height=550,
        plot_bgcolor=theme["plot_bg"],
        paper_bgcolor=theme["plot_bg"],
        font={
            "family": "Segoe UI, Arial, sans-serif",
            "color": theme["text_color"],
            "size": 14,
        },
        xaxis=dict(
            gridcolor=theme["grid_color"],
            gridwidth=2,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color=theme["text_color"]),
            title_font=dict(size=16, weight="bold", color=theme["text_color"]),
        ),
        yaxis=dict(
            gridcolor=theme["grid_color"],
            gridwidth=2,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color=theme["text_color"]),
            title_font=dict(size=16, weight="bold", color=theme["text_color"]),
        ),
        legend=dict(
            bgcolor=f"rgba({255 if not dark_mode else 45},{255 if not dark_mode else 45},{255 if not dark_mode else 45},0.9)",
            bordercolor="#bdc3c7" if not dark_mode else "#555555",
            borderwidth=2,
            font=dict(size=12, color=theme["text_color"]),
        ),
        margin=dict(l=80, r=80, t=100, b=80),
    )

    return fig


@callback(
    [Output("plots-container", "children"), Output("plots-container", "style")],
    [
        Input("technique-dropdown", "value"),
        Input("vmi-dropdown", "value"),
        Input("insert-dropdown", "value"),
        Input("plot-selection", "value"),
        Input("options-toggle", "value"),
    ],
)
def update_plots(techniques, vmis, inserts, selected_plots, options):
    """Update plots based on filter selections"""
    filtered_frames = filter_frames(techniques or [], vmis or [], inserts or [])
    show_bands = "show_bands" in (options or [])
    smooth_lsf = "smooth_lsf" in (options or [])
    dark_mode = "dark_mode" in (options or [])

    theme = get_theme_styles(dark_mode)
    plots = []

    plot_style = {
        "marginBottom": 30,
        "backgroundColor": theme["card_bg"],
        "borderRadius": "15px",
        "boxShadow": (
            "0 8px 25px rgba(0, 0, 0, 0.1)"
            if not dark_mode
            else "0 8px 25px rgba(0, 0, 0, 0.3)"
        ),
        "padding": "20px",
        "border": f"1px solid {theme['border_color']}",
    }

    if "lsf" in selected_plots:
        plots.append(
            html.Div(
                [
                    dcc.Graph(
                        figure=create_lsf_plot(
                            filtered_frames,
                            techniques,
                            vmis,
                            inserts,
                            smooth_lsf,
                            dark_mode,
                        ),
                    )
                ],
                style=plot_style,
            )
        )

    if "esf" in selected_plots:
        plots.append(
            html.Div(
                [
                    dcc.Graph(
                        figure=create_esf_plot(
                            filtered_frames, techniques, vmis, inserts, dark_mode
                        ),
                    )
                ],
                style=plot_style,
            )
        )

    if "contrast" in selected_plots:
        plots.append(
            html.Div(
                [
                    dcc.Graph(
                        figure=create_contrast_plot(
                            filtered_frames,
                            techniques,
                            vmis,
                            inserts,
                            show_bands,
                            dark_mode,
                        ),
                    )
                ],
                style=plot_style,
            )
        )

    container_style = {"backgroundColor": theme["bg_color"]}

    return plots, container_style


def load_frames_from_dataloader(data_loader):
    """Load frames from your DataLoader class"""
    global frames
    frames = data_loader.load_data()


if __name__ == "__main__":
    data_loader = DataLoader("data")
    load_frames_from_dataloader(data_loader)
    app.run(debug=True)
