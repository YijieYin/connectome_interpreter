import io
import pkgutil
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

DATA_SOURCES: dict[str, str] = {
    "DoOR_adult": "data/DoOR/processed_door_adult.csv",
    "DoOR_adult_sfr_subtracted": "data/DoOR/processed_door_adult_sfr_subtracted.csv",
    "Dweck_adult_chem": "data/Dweck2018/adult_chem2glom.csv",
    "Dweck_adult_fruit": "data/Dweck2018/adult_fruit2glom.csv",
    "Dweck_larva_chem": "data/Dweck2018/larva_chem2or.csv",
    "Dweck_larva_fruit": "data/Dweck2018/larva_fruit2or.csv",
    "Nern2024": "data/Nern2024/ME-columnar-cells-hex-location.csv",
    "Matsliah2024": "data/Matsliah2024/fafb_right_vis_cols.csv",
    "Badel2016_PN": "data/Badel2016/Badel2016.csv",
    "Zhao2024": "data/Zhao2024/ucl_hex_right_20240701_tomale.csv",
}


def load_dataset(dataset: str) -> pd.DataFrame:
    """
    Load the dataset from the package data folder. These datasets have been
    preprocessed to work with connectomics data. The preprocessing scripts are
    in this repository: https://github.com/YijieYin/interpret_connectome.

    Args:
        dataset: (str) The name of the dataset to load. Options are:

            - 'DoOR_adult': mapping from glomeruli to chemicals, from Munch and Galizia DoOR dataset (https://www.nature.com/articles/srep21841).
            - 'DoOR_adult_sfr_subtracted': mapping from glomeruli to chemicals, with spontaneous firing rate subtracted. There are therefore negative values.
            - 'Dweck_adult_chem': mapping from glomeruli to chemicals extracted from fruits, from Dweck et al. 2018 (https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6). Firing rates normalised to between 0 and 1.
            - 'Dweck_adult_fruit': mapping from glomeruli to fruits, from Dweck et al. 2018. Number of responses normalised to between 0 and 1.
            - 'Dweck_larva_chem': mapping from olfactory receptors to chemicals, from Dweck et al. 2018. Firing rates normalised to between 0 and 1.
            - 'Dweck_larva_fruit': mapping from olfactory receptors to fruits from Dweck et al. 2018. Number of responses normalised to between 0 and 1.
            - 'Nern2024': columnar coordinates of individual cells from a collection of columnar cell types within the medulla of the right optic lobe, from Nern et al. 2024 (https://www.biorxiv.org/content/10.1101/2024.04.16.589741v2).
            - 'Matsliah2024': columnar coordinates of individual cells from a collection of columnar cell types in the right optic lobe from FAFB, from Matsliah et al. 2024 (https://www.nature.com/articles/s41586-024-07981-1).
            - 'Badel2016_PN': mapping from olfactory projection neurons to odours, from Badel et al. 2016 (https://www.cell.com/neuron/fulltext/S0896-6273(16)30201-X).
            - 'Zhao2024': mapping from hexagonal coordinates to 3D coordinates, update from Zhao et al. 2022 (https://www.biorxiv.org/content/10.1101/2022.12.14.520178v1).

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame. For the adult, the glomeruli
        are in the rows. For the larva, receptors are in the rows.
    """

    try:
        data = pkgutil.get_data("connectome_interpreter", DATA_SOURCES[dataset])
    except KeyError as exc:
        raise ValueError(
            "Dataset not recognized. Please choose from {}".format(
                list(DATA_SOURCES.keys())
            )
        ) from exc

    return pd.read_csv(io.BytesIO(data), index_col=0)


def map_to_experiment(df, dataset=None, custom_experiment=None):
    """
    Map the connectomics data to experimental data. For example, if odour1
    excites neuron1 0.5, and neuron2 0.6; both neuron1 and neuron2 output to
    neuron3 (0.7 and 0.8 respectively), then the output of neuron3 to odour1
    is 0.5*0.7 + 0.6*0.8 = 0.83. The result would only be 1 if a stimulus
    excites neurons 100%, and those neurons constitue 100% of the downstream
    neuron's input.

    Args:
        df (pd.DataFrame): The connectivity data. Standardised input (e.g. glomeruli,
            receptors) in rows, observations (target neurons) in columns.
        dataset (str): The name of the dataset to load. Options are:

            - 'DoOR_adult': mapping from glomeruli to chemicals, from Munch and Galizia DoOR dataset (https://www.nature.com/articles/srep21841).
            - 'DoOR_adult_sfr_subtracted': mapping from glomeruli to chemicals, with spontaneous firing rate subtracted. There are therefore negative values.
            - 'Dweck_adult_chem': mapping from glomeruli to chemicals extracted from fruits, from Dweck et al. 2018 (https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6). Firing rates normalised to between 0 and 1.
            - 'Dweck_adult_fruit': mapping from glomeruli to fruits, from Dweck et al. 2018. Number of responses normalised to between 0 and 1.
            - 'Dweck_larva_chem': mapping from olfactory receptors to chemicals, from Dweck et al. 2018. Firing rates normalised to between 0 and 1.
            - 'Dweck_larva_fruit': mapping from olfactory receptors to fruits, from Dweck et al. 2018. Number of responses normalised to between 0 and 1.
            - 'Nern2024': columnar coordinates of individual cells from a collection of columnar cell types within the medulla of the right optic lobe, from Nern et al. 2024.
            - 'Badel2016_PN': mapping from olfactory projection neurons to odours, from Badel et al. 2016 (https://www.cell.com/neuron/fulltext/S0896-6273(16)30201-X).

        custom_experiment (pd.DataFrame): A custom experimental dataset to compare the
            connectomics data to. The row indices of this dataframe must match the row
            indices of df. They are the units of comparison (e.g. glomeruli).

    Returns:
        pd.DataFrame: The similarity between the connectomics data and the experimental
        data. Rows are neurons, columns are external stimulus.
    """

    # try:
    #     from sklearn.metrics.pairwise import cosine_similarity
    # except ImportError as e:
    #     raise ImportError(
    #         "To use this function, please install scikit-learn. You can
    # install it with 'pip install scikit-learn'.") from e
    if dataset is not None and custom_experiment is not None:
        raise ValueError(
            "Please provide either a dataset or a custom_experiment, not both."
        )
    if dataset is None and custom_experiment is None:
        raise ValueError("Please provide either a dataset or a custom_experiment.")
    if dataset is not None:
        data = load_dataset(dataset)
    else:
        data = custom_experiment

    # take the intersection of glomeruli
    data = data[data.index.isin(df.index)]
    df_intersect = df[df.index.isin(data.index)]
    df_intersect = df_intersect.reindex(data.index)

    # multiply the correpsonding values using matmul
    target2chem = np.dot(df_intersect.values.T, data.values)
    # Assign appropriate column names
    target2chem = pd.DataFrame(
        target2chem, index=df_intersect.columns, columns=data.columns
    )
    return target2chem


def hex_heatmap(
    df: pd.Series | pd.DataFrame,
    style: dict | None = None,
    sizing: dict | None = None,
    dpi: int = 72,
    custom_colorscale: list | None = None,
    global_min: float | None = None,
    global_max: float | None = None,
    dataset: str | None = "mcns_right",
) -> go.Figure:
    """
    Generate a hexagonal heat map plot of the data. The index of the data
    should be formatted as strings of the form '-12,34', where the first
    number is the x-coordinate and the second number is the y-coordinate.

    Args:
        df : pd.Series | pd.DataFrame
            The data to plot. Each column will generate a separate frame in
            the plot.
        style : dict, default=None
            Dict containing styling formatting variables. Possible keys are:

                - 'font_type': str, default='arial'
                - 'linecolor': str, default='black'
                - 'papercolor': str, default='rgba(255,255,255,255)' (white)

        sizing : dict, default=None
            Dict containing size formatting variables. Possible keys are:

                - 'fig_width': int, default=260 (mm)
                - 'fig_height': int, default=220 (mm)
                - 'fig_margin': int, default=0 (mm)
                - 'fsize_ticks_pt': int, default=20 (points)
                - 'fsize_title_pt': int, default=20 (points)
                - 'markersize': int, default=18 if dataset='mcns_right', 20 if dataset='fafb_right'
                - 'ticklen': int, default=15
                - 'tickwidth': int, default=5
                - 'axislinewidth': int, default=3
                - 'markerlinewidth': int, default=0.9
                - 'cbar_thickness': int, default=20
                - 'cbar_len': float, default=0.75

        dpi : int, default=72
            Dots per inch for the output figure. Standard is 72 for screen/SVG/PDF.
            Use higher values (e.g., 300) for print-quality output.
        custom_colorscale : list, default=None
            Custom colorscale for the heatmap. If None, defaults to white-to-blue
            colorscale [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]].
        global_min : float, default=None
            Global minimum value for the color scale.
            If None, the minimum value of the data is used but if that is negative, use 0.
        global_max : float, default=None
            Global maximum value for the color scale.
            If None, the maximum value of the data is used.
        dataset : str, default='mcns_right'
            The dataset to use for the hexagon locations. Options are:

                - 'mcns_right': columnar coordinates of individual cells from columnar cell types: L1, L2, L3, L5, Mi1, Mi4, Mi9, C2, C3, Tm1, Tm2, Tm4, Tm9, Tm20, T1, within the medulla of the right optic lobe, from Nern et al. 2024.
                - 'fafb_right': columnar coordinates of individual cells from columnar cell types, in the right optic lobe of FAFB, from Matsliah et al. 2024.

    Returns:
        fig : go.Figure
    """

    def bg_hex():
        """
        Generate a scatter plot of the background hexagons."
        """
        goscatter = go.Scatter(
            x=background_hex["x"],
            y=background_hex["y"],
            mode="markers",
            marker_symbol=symbol_number,
            marker={
                "size": sizing["markersize"],
                "color": "white",
                "line": {
                    "width": sizing["markerlinewidth"],
                    "color": "lightgrey",
                },
            },
            showlegend=False,
        )
        return goscatter

    def data_hex(aseries):
        """
        Generate a scatter plot of the data hexagons.""
        """
        goscatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker_symbol=symbol_number,
            customdata=np.stack([x_vals, y_vals, aseries.values], axis=-1),
            hovertemplate="x: %{customdata[0]}<br>y: %{customdata[1]}<br>effective weight: %{customdata[2]:.4f}",
            marker={
                "cmin": global_min,
                "cmax": global_max,
                "size": sizing["markersize"],
                "color": aseries.values,
                "line": {
                    "width": sizing["markerlinewidth"],
                    "color": "lightgrey",
                },
                "colorbar": {
                    "orientation": "v",
                    "outlinecolor": style["linecolor"],
                    "outlinewidth": sizing["axislinewidth"],
                    "thickness": sizing["cbar_thickness"],
                    "len": sizing["cbar_len"],
                    "tickmode": "array",
                    "ticklen": sizing["ticklen"],
                    "tickwidth": sizing["tickwidth"],
                    "tickcolor": style["linecolor"],
                    "tickfont": {
                        "size": fsize_ticks_px,
                        "family": style["font_type"],
                        "color": style["linecolor"],
                    },
                    "tickformat": ".5f",
                    "title": {
                        "font": {
                            "family": style["font_type"],
                            "size": fsize_title_px,
                            "color": style["linecolor"],
                        },
                        "side": "right",
                    },
                },
                "colorscale": custom_colorscale,
            },
            showlegend=False,
        )
        return goscatter

    # Default styling and sizing parameters to use if not specified.
    default_style = {
        "font_type": "arial",
        "markerlinecolor": "rgba(0,0,0,0)",  # transparent
        "linecolor": "black",
        "papercolor": "rgba(255,255,255,255)",
    }

    if dataset == "mcns_right":
        markersize = 18
    elif dataset == "fafb_right":
        markersize = 20
    else:
        # raise error
        raise ValueError(
            "Dataset not recognized. Currently available datasets are 'mcns_right', "
            "'fafb_right'."
        )

    default_sizing = {
        "fig_width": 260,  # units = mm
        "fig_height": 220,  # units = mm
        "fig_margin": 0,
        "fsize_ticks_pt": 20,
        "fsize_title_pt": 20,
        "markersize": markersize,
        "ticklen": 15,
        "tickwidth": 5,
        "axislinewidth": 3,
        "markerlinewidth": 0.5,  # 0.9,
        "cbar_thickness": 20,
        "cbar_len": 0.75,
    }

    # If style is provided, update default_style with user values
    if style is not None:
        default_style.update(style)
    style = default_style

    if sizing is not None:
        default_sizing.update(sizing)
    sizing = default_sizing

    # Constants for unit conversion
    POINTS_PER_INCH = 72  # Typography standard: 1 point = 1/72 inch
    MM_PER_INCH = 25.4  # Standard conversion: 1 inch = 25.4 mm

    # sizing of the figure and font
    pixelsperinch = dpi  # Use the provided DPI value
    pixelspermm = pixelsperinch / MM_PER_INCH

    # Default colorscale
    if custom_colorscale is None:
        custom_colorscale = [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]]

    area_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    area_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm

    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch

    # Get global min and max for consistent color scale
    # minimum of 0 and df.values.min()
    vals = df.to_numpy()
    if global_min is None:
        global_min = min(0, vals.min())
    if global_max is None:
        global_max = vals.max()

    # Symbol number to choose to plot hexagons
    symbol_number = 15

    # load all hex coordinates
    if dataset == "mcns_right":
        background_hex = load_dataset("Nern2024")
    elif dataset == "fafb_right":
        background_hex = load_dataset("Matsliah2024")
    else:
        # raise error
        raise ValueError(
            "Dataset not recognized. Currently available datasets are 'mcns_right', "
            "'fafb_right'."
        )
    # only get the unique combination of 'x' and 'y' columns
    background_hex = background_hex.drop_duplicates(subset=["x", "y"])

    # initiate plot
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        height=area_height,
        width=area_width,
        margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
        paper_bgcolor=style["papercolor"],
        plot_bgcolor=style["papercolor"],
    )
    fig.update_xaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )

    # Convert index values (formatted as '-12,34') into separate x and y coordinates
    df = df[(df.index != "nan") & (~df.index.isnull())]
    coords = [tuple(map(float, idx.split(","))) for idx in df.index]
    x_vals, y_vals = zip(*coords)  # Separate into x and y lists

    if isinstance(df, pd.Series) or len(df.columns) == 1:
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        fig.add_trace(bg_hex())
        fig.add_trace(data_hex(df))

    elif isinstance(df, pd.DataFrame):
        # Adjust figure size - add extra height for slider
        slider_height = 100  # pixels
        area_height += slider_height

        # Create frames for slider
        frames = []
        slider_steps = []

        # Add base layout
        fig.update_layout(
            autosize=False,
            height=area_height,
            width=area_width,
            margin={
                "l": 0,
                "r": 0,
                "b": slider_height,
                "t": 0,
                "pad": 0,
            },  # Add bottom margin for slider
            paper_bgcolor=style["papercolor"],
            plot_bgcolor=style["papercolor"],
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {
                        "font": {"size": 16},
                        "visible": True,
                        "xanchor": "right",
                    },
                    "pad": {"b": 10, "t": 0},  # Adjusted padding
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,  # Move slider below plot
                    "steps": [],
                }
            ],
        )

        # Create frames for each column
        for i, col_name in enumerate(df.columns):
            series = df[col_name]
            frame_data = [
                bg_hex(),
                data_hex(series),
            ]

            frames.append(go.Frame(data=frame_data, name=str(i)))

            # Add to slider
            slider_steps.append(
                {
                    "args": [
                        [str(i)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                    "label": col_name,
                    "method": "animate",
                }
            )

            # Set initial display to first column
            if i == 0:
                fig.add_traces(frame_data)

        # Update slider with all steps
        fig.layout.sliders[0].steps = slider_steps
        fig.frames = frames

        # Update axes
        fig.update_xaxes(
            showgrid=False, showticklabels=False, showline=False, visible=False
        )
        fig.update_yaxes(
            showgrid=False, showticklabels=False, showline=False, visible=False
        )

    else:
        # raise error
        raise ValueError("df must be a pd.Series or pd.DataFrame")

    return fig


def looming_stimulus(start_coords, all_coords, n_time=4):
    """
    Generate a list of lists of coordinates for a looming stimulus. The stimulus starts
    at the start_coords and expands outwards in a hexagonal pattern. The stimulus
    expands for n_time steps. Currently the expansion happens one layer at a time.

    Args:
        start_coords (list): List of strings of the form 'x,y' where x and y are the
            coordinates of the starting hexes for the stimulus.
        all_coords (list): List of strings of the form 'x,y' where x and y are the
            coordinates of all hexes in the grid.
        n_time (int): Default=4. Number of time steps for the stimulus to expand.

    Returns:
        stim_str (list): List of lists of strings of the form 'x,y' where x and y are
        the coordinates of the hexes that are stimulated at each time step.
    """
    coords = [tuple(map(float, idx.split(","))) for idx in all_coords]
    x_vals, y_vals = zip(*coords)  # Separate into x and y lists

    # sort and rank x_vals
    x_sorted = sorted(list(set(x_vals)))
    x_to_rank = {x: rank for rank, x in enumerate(x_sorted)}
    rank_to_x = {rank: x for rank, x in enumerate(x_sorted)}
    y_sorted = sorted(list(set(y_vals)))
    y_to_rank = {y: rank for rank, y in enumerate(y_sorted)}
    rank_to_y = {rank: y for rank, y in enumerate(y_sorted)}

    start = [tuple(map(float, idx.split(","))) for idx in start_coords]
    stimulus = []
    stimulus.append(start)
    for atime in range(n_time):
        for x, y in start:
            start_copy = start.copy()
            # hexes above and below x
            if y_to_rank[y] + 2 in rank_to_y:
                start_copy.append((x, rank_to_y[y_to_rank[y] + 2]))
            if y_to_rank[y] - 2 in rank_to_y:
                start_copy.append((x, rank_to_y[y_to_rank[y] - 2]))
            # hexes to the left
            if x_to_rank[x] + 1 in rank_to_x:
                if y_to_rank[y] + 1 in rank_to_y:
                    start_copy.append(
                        (rank_to_x[x_to_rank[x] + 1], rank_to_y[y_to_rank[y] + 1])
                    )
                if y_to_rank[y] - 1 in rank_to_y:
                    start_copy.append(
                        (rank_to_x[x_to_rank[x] + 1], rank_to_y[y_to_rank[y] - 1])
                    )
            # hexes to the right
            if x_to_rank[x] - 1 in rank_to_x:
                if y_to_rank[y] + 1 in rank_to_y:
                    start_copy.append(
                        (rank_to_x[x_to_rank[x] - 1], rank_to_y[y_to_rank[y] + 1])
                    )
                if y_to_rank[y] - 1 in rank_to_y:
                    start_copy.append(
                        (rank_to_x[x_to_rank[x] - 1], rank_to_y[y_to_rank[y] - 1])
                    )

            start = list(set(start_copy))
        stimulus.append(start)

    stim_str = []
    for atime in range(n_time):
        stim_atime = []
        for x, y in stimulus[atime]:
            # Format x and y to remove .0 if they're integers
            x_str = str(int(x)) if x == int(x) else str(x)
            y_str = str(int(y)) if y == int(y) else str(y)
            stim_atime.append(f"{x_str},{y_str}")
        stim_str.append(stim_atime)
    return stim_str


def make_sine_stim(phase=0, amplitude=1, n=8):
    """
    Generate a dictionary of values representing a sine wave stimulus with a given phase
    and amplitude. The sine wave is defined over n points, starting from the given phase.

    Args:
        phase (int): Phase of the sine wave in degrees. Default is 0.
        amplitude (float): Amplitude of the sine wave. Default is 1.
        n (int): Number of points in the sine wave. Default is 8.

    Returns:
        dict: A dictionary where keys are indices from 1 to n, and values are the
        corresponding sine wave values.
    """
    x = (phase % 180) / 180 * np.pi
    x = np.linspace(x, x + np.pi, n)
    y = amplitude * abs(np.sin(x))
    return dict(zip(range(1, n + 1), y))


def plot_mollweide_projection(
    data: pd.Series | pd.DataFrame,
    fig_size: tuple = (900, 700),
    custom_colorscale: list | None = None,
    global_min: float | None = None,
    global_max: float | None = None,
    dataset: str = "Zhao2024",
    marker_size: int = 8,
) -> go.Figure:
    """
    Generates a heatmap to visualize the value of column features per column using the
    mollweide projection.

    Args:
        data (pd.Series | pd.DataFrame): Data with index formatted as strings of the
            form '-12,34', where the first number is the x-coordinate and the second
            number is the y-coordinate. The data to plot. Each column will generate a
            separate frame in the plot.
        fig_size (tuple): Size of the figure in pixels (width, height).
        custom_colorscale (str): Name of the Plotly colorscale to use.
        global_min (float | None): Global minimum value for the color scale. If this
            minumum is >0, 0 is used.
        global_max (float | None): Global maximum value for the color scale. If None,
            the maximum value of the data is used.
        dataset (str): The dataset to use for the hexagon locations. Options are:

            - 'Zhao2024': mapping from hexagonal coordinates to 3D coordinates, update from Zhao et al. 2022 (https://www.biorxiv.org/content/10.1101/2022.12.14.520178v1).

        marker_size (int): Size of markers in the plot.

    Returns:
        go.Figure: A Plotly figure object containing the mollweide projection heatmap.
    """

    def cart2sph(xyz: np.array) -> np.array:
        """
        Convert Cartesian to spherical coordinates.
        Theta is polar angle (from +z), phi is angle from +x to +y.
        """
        r = np.sqrt((xyz**2).sum(1))
        theta = np.arccos(xyz[:, 2])
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi[phi < 0] = phi[phi < 0] + 2 * np.pi
        return np.stack((r, theta, phi), axis=1)

    def sph2Mollweide(thetaphi: np.array) -> np.array:
        """
        Spherical (viewed from outside) to Mollweide,
            cf. https://mathworld.wolfram.com/MollweideProjection.html
        """
        azim = thetaphi[:, 1]
        azim[azim > np.pi] = azim[azim > np.pi] - 2 * np.pi  # longitude/azimuth
        elev = np.pi / 2 - thetaphi[:, 0]  # lattitude/elevation in radian

        N = len(azim)  # number of points
        xy = np.zeros((N, 2))  # output
        for i in range(N):
            theta = np.arcsin(2 * elev[i] / np.pi)
            if np.abs(np.abs(theta) - np.pi / 2) < 0.001:
                xy[i,] = [
                    2 * np.sqrt(2) / np.pi * azim[i] * np.cos(theta),
                    np.sqrt(2) * np.sin(theta),
                ]
            else:
                # to calculate theta
                dtheta = 1
                while dtheta > 1e-3:
                    theta_new = theta - (
                        2 * theta + np.sin(2 * theta) - np.pi * np.sin(elev[i])
                    ) / (2 + 2 * np.cos(2 * theta))
                    dtheta = np.abs(theta_new - theta)
                    theta = theta_new
                xy[i,] = [
                    2 * np.sqrt(2) / np.pi * azim[i] * np.cos(theta),
                    np.sqrt(2) * np.sin(theta),
                ]
        return xy

    def create_mollweide_guidelines():
        """
        Create Mollweide projection guidelines as plotly traces
        """
        traces = []

        # Create meridians
        ww = np.stack((np.linspace(0, 180, 19), np.repeat(-180, 19)), axis=1)
        w = np.stack((np.linspace(180, 0, 19), np.repeat(-90, 19)), axis=1)
        m = np.stack((np.linspace(0, 180, 19), np.repeat(0, 19)), axis=1)
        e = np.stack((np.linspace(180, 0, 19), np.repeat(90, 19)), axis=1)
        ee = np.stack((np.linspace(0, 180, 19), np.repeat(180, 19)), axis=1)
        pts = np.vstack((ww, w, m, e, ee))
        rtp = np.insert(pts / 180 * np.pi, 0, np.repeat(1, pts.shape[0]), axis=1)
        meridians_xy = sph2Mollweide(rtp[:, 1:3])

        traces.append(
            go.Scatter(
                x=meridians_xy[:, 0],
                y=meridians_xy[:, 1],
                mode="lines",
                line=dict(color="lightgrey", width=0.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Create parallels
        for lat in [45, 90, 135]:
            pts = np.stack((np.repeat(lat, 37), np.linspace(-180, 180, 37)), axis=1)
            rtp = np.insert(pts / 180 * np.pi, 0, np.repeat(1, pts.shape[0]), axis=1)
            parallel_xy = sph2Mollweide(rtp[:, 1:3])

            traces.append(
                go.Scatter(
                    x=parallel_xy[:, 0],
                    y=parallel_xy[:, 1],
                    mode="lines",
                    line=dict(color="lightgrey", width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return traces

    def create_data_scatter(series_data, x_coords, y_coords, column_name=None):
        """Create scatter plot for data points"""
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(
                color=series_data.values,
                colorscale=custom_colorscale,
                cmin=global_min,
                cmax=global_max,
                size=marker_size,
                colorbar=dict(
                    title=dict(text="effective weight", side="right"),
                ),
            ),
            customdata=np.stack([x_coords, y_coords, series_data.values], axis=-1),
            hovertemplate="x: %{customdata[0]:.2f}<br>y: %{customdata[1]:.2f}<br>effective weight: %{customdata[2]:.4f}<extra></extra>",
            showlegend=False,
        )

    # Default colorscale
    if custom_colorscale is None:
        custom_colorscale = [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]]

    # Clean data - remove NaN indices
    data = data[(data.index != "nan") & (~data.index.isnull())]

    # Convert string indices to coordinate arrays
    coords = [tuple(map(float, idx.split(","))) for idx in data.index]
    coord_array = np.array(coords)

    # Get global min and max for consistent color scale
    vals = data.to_numpy() if isinstance(data, pd.DataFrame) else data.values
    if global_min is None:
        global_min = min(0, vals.min())
    if global_max is None:
        global_max = vals.max()

    # Load eyemap data and convert coordinates
    ucl_hex = load_dataset(dataset)
    rtp2 = cart2sph(ucl_hex[["x", "y", "z"]].values)
    xy = sph2Mollweide(rtp2[:, 1:3])
    xy[:, 0] = -xy[:, 0]  # flip x axis
    xypq_moll = np.concatenate((xy, ucl_hex[["p", "q"]].values), axis=1)
    xypq_moll = pd.DataFrame(xypq_moll, columns=["x", "y", "p", "q"])
    xypq_moll[["p", "q"]] = xypq_moll[["p", "q"]].astype(int)

    # Convert data coordinates to Mollweide
    hex1_id = (coord_array[:, 1] - coord_array[:, 0]) / 2
    hex2_id = (coord_array[:, 1] + coord_array[:, 0]) / 2

    coord_df = pd.DataFrame({"hex1_id": hex1_id, "hex2_id": hex2_id}, index=data.index)

    merged_coords = coord_df.merge(
        xypq_moll, left_on=["hex1_id", "hex2_id"], right_on=["q", "p"], how="left"
    )

    x_mollweide = merged_coords["x"].values
    y_mollweide = merged_coords["y"].values

    # Create figure
    fig = go.Figure()

    # Add guidelines
    guidelines = create_mollweide_guidelines()
    for trace in guidelines:
        fig.add_trace(trace)

    # Handle single series vs DataFrame
    if isinstance(data, pd.Series) or (
        isinstance(data, pd.DataFrame) and len(data.columns) == 1
    ):
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]

        # Single plot
        fig.add_trace(create_data_scatter(data, x_mollweide, y_mollweide))

    elif isinstance(data, pd.DataFrame):
        # Multiple columns - create frames for slider
        frames = []
        slider_steps = []

        # Create frames for each column
        for i, col_name in enumerate(data.columns):
            series = data[col_name]

            # Create frame data (guidelines + data scatter)
            frame_traces = guidelines + [
                create_data_scatter(series, x_mollweide, y_mollweide)
            ]
            frames.append(go.Frame(data=frame_traces, name=str(i)))

            # Add slider step
            slider_steps.append(
                {
                    "args": [
                        [str(i)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                    "label": col_name,
                    "method": "animate",
                }
            )

            # Set initial display to first column
            if i == 0:
                fig.add_trace(
                    create_data_scatter(series, x_mollweide, y_mollweide, col_name)
                )

        # Add frames to figure
        fig.frames = frames

        # Add slider
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {
                        "font": {"size": 16},
                        "visible": True,
                        "xanchor": "right",
                    },
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": slider_steps,
                }
            ]
        )

    # Update layout
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis=dict(
            range=[-np.pi, np.pi],
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            showticklabels=False,
            showline=False,
            visible=False,
        ),
        yaxis=dict(
            range=[-np.pi / 2, np.pi / 2],
            showgrid=False,
            showticklabels=False,
            showline=False,
            visible=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=50 if isinstance(data, pd.DataFrame) and len(data.columns) > 1 else 0,
        ),
    )

    return fig
