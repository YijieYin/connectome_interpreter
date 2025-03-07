import io
import pkgutil

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_dataset(dataset):
    """
    Load the dataset from the package data folder. These datasets have been
    preprocessed to work with connectomics data. The preprocessing scripts are
    in this repository: https://github.com/YijieYin/interpret_connectome.

    Args:
        dataset : str
            The name of the dataset to load. Options are:

            - 'DoOR_adult': mapping from glomeruli to chemicals, from Munch
                and Galizia DoOR dataset (https://www.nature.com/articles/srep21841).
            - 'DoOR_adult_sfr_subtracted': mapping from glomeruli to chemicals,
                with spontaneous firing rate subtracted. There are therefore
                negative values.
            - 'Dweck_adult_chem': mapping from glomeruli to chemicals
                extracted from fruits, from Dweck et al. 2018
                (https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6).
                Firing rates normalised to between 0 and 1.
            - 'Dweck_adult_fruit': mapping from glomeruli to fruits, from
                Dweck et al. 2018. Number of responses normalised to between 0
                and 1.
            - 'Dweck_larva_chem': mapping from olfactory receptors to
                chemicals, from Dweck et al. 2018. Firing rates normalised to
                between 0 and 1.
            - 'Dweck_larva_fruit': mapping from olfactory receptors to fruits,
                from Dweck et al. 2018. Number of responses normalised to
                between 0 and 1.
            - 'Nern2024': columnar coordinates of individual cells from a 
                collection of columnar cell types within the medulla of the 
                right optic lobe, from Nern et al. 2024.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame. For the adult, the
            glomeruli are in the rows. For the larva, receptors are in the
            rows.
    """

    if dataset == "DoOR_adult":
        data = pkgutil.get_data(
            "connectome_interpreter", "data/DoOR/processed_door_adult.csv"
        )
    elif dataset == "DoOR_adult_sfr_subtracted":
        data = pkgutil.get_data(
            "connectome_interpreter",
            "data/DoOR/processed_door_adult_sfr_subtracted.csv",
        )
    elif dataset == "Dweck_adult_chem":
        data = pkgutil.get_data(
            "connectome_interpreter", "data/Dweck2018/adult_chem2glom.csv"
        )
    elif dataset == "Dweck_adult_fruit":
        data = pkgutil.get_data(
            "connectome_interpreter", "data/Dweck2018/adult_fruit2glom.csv"
        )
    elif dataset == "Dweck_larva_chem":
        data = pkgutil.get_data(
            "connectome_interpreter", "data/Dweck2018/larva_chem2or.csv"
        )
    elif dataset == "Dweck_larva_fruit":
        data = pkgutil.get_data(
            "connectome_interpreter", "data/Dweck2018/larva_fruit2or.csv"
        )
    elif dataset == "Nern2024":
        data = pkgutil.get_data(
            "connectome_interpreter", "data/Nern2024/ME-columnar-cells-hex-location.csv"
        )
    else:
        raise ValueError(
            "Dataset not recognized. Please choose from 'DoOR_adult', 'DoOR_adult_sfr_subtracted', 'Dweck_adult_chem', 'Dweck_adult_fruit', 'Dweck_larva_chem', 'Dweck_larva_fruit', 'Nern2024'."
        )

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
        df : pd.DataFrame
            The connectivity data. Standardised input (e.g. glomeruli,
            receptors) in rows, observations (target neurons) in columns.
        dataset : str
            The name of the dataset to load. Options are:

            - 'DoOR_adult': mapping from glomeruli to chemicals, from Munch
                and Galizia DoOR dataset
                (https://www.nature.com/articles/srep21841).
            - 'DoOR_adult_sfr_subtracted': mapping from glomeruli to chemicals,
                with spontaneous firing rate subtracted. There are therefore
                negative values.
            - 'Dweck_adult_chem': mapping from glomeruli to chemicals
                extracted from fruits, from Dweck et al. 2018
                (https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6).
                Firing rates normalised to between 0 and 1.
            - 'Dweck_adult_fruit': mapping from glomeruli to fruits, from
                Dweck et al. 2018. Number of responses normalised to between
                0 and 1.
            - 'Dweck_larva_chem': mapping from olfactory receptors to
                chemicals, from Dweck et al. 2018. Firing rates normalised to
                between 0 and 1.
            - 'Dweck_larva_fruit': mapping from olfactory receptors to fruits,
                from Dweck et al. 2018. Number of responses normalised to
                between 0 and 1.
            - 'Nern2024': columnar coordinates of individual cells from a 
                collection of columnar cell types within the medulla of the 
                right optic lobe, from Nern et al. 2024.
        custom_experiment : pd.DataFrame
            A custom experimental dataset to compare the connectomics data to.
            The row indices of this dataframe must match the row indices of df.
            They are the units of comparison (e.g. glomeruli).

    Returns:
        pd.DataFrame: The similarity between the connectomics data and the
            experimental data. Rows are neurons, columns are external stimulus.
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
        raise ValueError(
            "Please provide either a dataset or a custom_experiment."
        )
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
    df:pd.Series,
    style:dict=None,
    sizing:dict=None,
) -> go.Figure:
    """
    Generate a hexagonal heat map plot of the data in a pandas series 'df'.

    Args: 
        df : pd.Series
            A Series where the index is formatted as 'x,y' coordinates and values represent data to plot.
        style : dict, default=None
            Dict containing styling formatting variables.
        sizing : dict, default=None
            Dict containing size formatting variables.

    Returns:
        fig : go.Figure
    """
    # Default styling and sizing parameters to use if not specified.
    if style is None:
        style = {
            "font_type": "arial",
            "markerlinecolor": "rgba(0,0,0,0)", #transparent
            "linecolor": "black",
            "papercolor": "rgba(255,255,255,255)"
        }

    if sizing is None:
        sizing = {
            "fig_width": 260,  # units = mm
            "fig_height": 220,  # units = mm
            "fig_margin": 0,
            "fsize_ticks_pt": 20,
            "fsize_title_pt": 20,
            "markersize": 18,
            "ticklen": 15,
            "tickwidth": 5,
            "axislinewidth": 3,
            "markerlinewidth": 0.9,
            "cbar_thickness": 20,
            "cbar_len": 0.75,
        }

    # sizing of the figure and font
    pixelsperinch = 72 # for svg and pdf
    pixelspermm = pixelsperinch / 25.4
    area_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    area_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / 72) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / 72) * pixelsperinch

    # Convert index values (formatted as '-12,34') into separate x and y coordinates
    coords = [tuple(map(int, idx.split(","))) for idx in df.index]
    x_vals, y_vals = zip(*coords)  # Separate into x and y lists

    # initiate plot
    fig = go.Figure()
    fig.update_layout(
        autosize=False
      , height=area_height
      , width=area_width
      , margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0}
      , paper_bgcolor=style["papercolor"]
      , plot_bgcolor=style["papercolor"]
    )
    fig.update_xaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    # Symbol number to choose to plot hexagons
    symbol_number = 15

    # Get the coordinates of all columns in the medulla:
    col_coords = pd.read_csv("../connectome_interpreter/data/Nern2024/ME-column-coords.csv")
    
    # Add empty white 'background' hexagons - all neuropils
    fig.add_trace(
        go.Scatter(
            x=col_coords["x"]
          , y=col_coords["y"]
          , mode="markers"
          , marker_symbol=symbol_number
          , marker={
                "size": sizing["markersize"]
              , "color": "white"
              , "line": {"width": sizing["markerlinewidth"], "color": "lightgrey"}
            }
          , showlegend=False
        )
    )

    custom_colorscale = [
    [0, 'rgb(255, 255, 255)']
    , [1, 'rgb(0, 20, 200)']
    ]

    # Add data
    fig.add_trace(
        go.Scatter(
            x=x_vals
          , y=y_vals
          , mode="markers"
          , marker_symbol=symbol_number
          , marker={
              "cmin": 0
              , "cmax": df.values.max()
              , "size": sizing["markersize"]
              , "color": df.values
              , "line": {
                    "width": sizing["markerlinewidth"]
                  , "color": style["markerlinecolor"]
                }
              , "colorbar": {
                    "orientation": "v"
                  , "outlinecolor": style["linecolor"]
                  , "outlinewidth": sizing["axislinewidth"]
                  , "thickness": sizing["cbar_thickness"]
                  , "len": sizing["cbar_len"]
                  , "tickmode": "array"
                  , "ticklen": sizing["ticklen"]
                  , "tickwidth": sizing["tickwidth"]
                  , "tickcolor": style["linecolor"]
                  , "tickfont": {
                        "size": fsize_ticks_px
                      , "family": style["font_type"]
                      , "color": style["linecolor"]
                    }
                  , "tickformat": ".5f"
                  , "title": {
                        "font": {
                            "family": style["font_type"]
                          , "size": fsize_title_px
                          , "color": style["linecolor"]
                        }
                      , "side": "right"
                    }
                }
              , "colorscale": custom_colorscale
            }
          , showlegend=False
        )
    )

    return fig

    # # or, cosine similarity
    # sim = cosine_similarity(df_intersect.T, data.T)
    # simdf = pd.DataFrame(sim, index=df_intersect.columns,
    #                      columns=data.columns)
    # return simdf

    # or, my custom similarity metric: element-wise multiplication of
    # connectivity and experimental activation
    # then divided by the number of non-zero elements in the experimental data:
    # # multiply the correpsonding values using matmul
    # target2chem = np.dot(df_intersect.values.T, data.values)
    # # Assign appropriate column names
    # target2chem = pd.DataFrame(
    #     target2chem, index=df_intersect.columns, columns=data.columns)
    # non_zero_counts = (data != 0).sum(axis=0)
    # # Normalize by the number of non-zero elements
    # # so that if a chemical only activates one receptor/glomerulus, it's not
    # discriminated against
    # # and if a neuron only connects to a subset of the receptors a chemical
    # excites, the number is punished
    # target2chem = target2chem.div(non_zero_counts, axis=1)
    # return target2chem
