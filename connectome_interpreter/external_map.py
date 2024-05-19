import pkgutil
import io

import pandas as pd


def load_dataset(dataset):
    '''
    Load the dataset from the package data folder. These datasets have been preprocessed to work with connectomics data. The preprocessing scripts are in this repository: https://github.com/YijieYin/interpret_connectome. 

    Args: 
        dataset : str
            The name of the dataset to load. Options are:

            - 'DoOR_adult': mapping from glomeruli to chemicals, from Munch and Galizia DoOR dataset (https://www.nature.com/articles/srep21841).  
            - 'DoOR_adult_sfr_subtracted': mapping from glomeruli to chemicals, with spontaneous firing rate subtracted. There are therefore negative values.  
            - 'Dweck_adult_chem': mapping from glomeruli to chemicals extracted from fruits, from Dweck et al. 2018 (https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6). Firing rates normalised to between 0 and 1.  
            - 'Dweck_adult_fruit': mapping from glomeruli to fruits, from Dweck et al. 2018. Number of responses normalised to between 0 and 1.  
            - 'Dweck_larva_chem': mapping from olfactory receptors to chemicals, from Dweck et al. 2018. Firing rates normalised to between 0 and 1.  
            - 'Dweck_larva_fruit': mapping from olfactory receptors to fruits, from Dweck et al. 2018. Number of responses normalised to between 0 and 1.  

    Returns: 
        pd.DataFrame: The dataset as a pandas DataFrame. For the adult, the glomeruli are in the rows. For the larva, receptors are in the rows. 
    '''

    if dataset == 'DoOR_adult':
        data = pkgutil.get_data('connectome_interpreter',
                                'data/DoOR/processed_door_adult.csv')
    elif dataset == 'DoOR_adult_sfr_subtracted':
        data = pkgutil.get_data(
            'connectome_interpreter', 'data/DoOR/processed_door_adult_sfr_subtracted.csv')
    elif dataset == 'Dweck_adult_chem':
        data = pkgutil.get_data('connectome_interpreter',
                                'data/Dweck2018/adult_chem2glom.csv')
    elif dataset == 'Dweck_adult_fruit':
        data = pkgutil.get_data('connectome_interpreter',
                                'data/Dweck2018/adult_fruit2glom.csv')
    elif dataset == 'Dweck_larva_chem':
        data = pkgutil.get_data('connectome_interpreter',
                                'data/Dweck2018/larva_chem2or.csv')
    elif dataset == 'Dweck_larva_fruit':
        data = pkgutil.get_data('connectome_interpreter',
                                'data/Dweck2018/larva_fruit2or.csv')

    return pd.read_csv(io.BytesIO(data), index_col=0)


def map_to_experiment(df, dataset=None, custom_experiment=None):
    '''
    Map the connectomics data to experimental data. Needs the scikit-learn package to calculate cosine similarity.

    Args:
        df : pd.DataFrame
            The connectivity data. Standardised input (e.g. glomeruli, receptors) in rows, observations (target neurons) in columns.
        dataset : str
            The name of the dataset to load. Options are:

            - 'DoOR_adult': mapping from glomeruli to chemicals, from Munch and Galizia DoOR dataset (https://www.nature.com/articles/srep21841).  
            - 'DoOR_adult_sfr_subtracted': mapping from glomeruli to chemicals, with spontaneous firing rate subtracted. There are therefore negative values.  
            - 'Dweck_adult_chem': mapping from glomeruli to chemicals extracted from fruits, from Dweck et al. 2018 (https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6). Firing rates normalised to between 0 and 1.  
            - 'Dweck_adult_fruit': mapping from glomeruli to fruits, from Dweck et al. 2018. Number of responses normalised to between 0 and 1.  
            - 'Dweck_larva_chem': mapping from olfactory receptors to chemicals, from Dweck et al. 2018. Firing rates normalised to between 0 and 1.  
            - 'Dweck_larva_fruit': mapping from olfactory receptors to fruits, from Dweck et al. 2018. Number of responses normalised to between 0 and 1.  
        custom_experiment : pd.DataFrame
            A custom experimental dataset to compare the connectomics data to. The row indices of this dataframe must match the row indices of df. They are the units of comparison (e.g. glomeruli). This function will output cosine similarity of the columns of the two dataframes. 

    Returns:
        pd.DataFrame: The similarity between the connectomics data and the experimental data. Rows are neurons, columns are chemicals or fruits. 
    '''

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise ImportError(
            "To use this function, please install scikit-learn. You can install it with 'pip install scikit-learn'.")
    if dataset is not None and custom_experiment is not None:
        raise ValueError(
            "Please provide either a dataset or a custom_experiment, not both.")
    if dataset is None and custom_experiment is None:
        raise ValueError(
            "Please provide either a dataset or a custom_experiment.")
    if dataset is not None:
        data = load_dataset(dataset)
    else:
        data = custom_experiment

    # take the intersection of glomeruli
    data = data[data.index.isin(df.index)]
    df_intersect = df[df.index.isin(data.index)]
    df_intersect = df_intersect.reindex(data.index)

    # cosine similarity
    sim = cosine_similarity(df_intersect.T, data.T)
    simdf = pd.DataFrame(sim, index=df_intersect.columns, columns=data.columns)

    # my custom similarity metric: element-wise multiplication of connectivity and experimental activation
    # then divided by the number of non-zero elements in the experimental data:
    # # multiply the correpsonding values using matmul
    # target2chem = np.dot(df_intersect.values.T, data.values)
    # # Assign appropriate column names
    # target2chem = pd.DataFrame(
    #     target2chem, index=df_intersect.columns, columns=data.columns)
    # non_zero_counts = (data != 0).sum(axis=0)
    # # Normalize by the number of non-zero elements
    # # so that if a chemical only activates one receptor/glomerulus, it's not discriminated against
    # # and if a neuron only connects to a subset of the receptors a chemical excites, the number is punished
    # target2chem = target2chem.div(non_zero_counts, axis=1)
    return simdf
