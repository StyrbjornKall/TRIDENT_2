import pandas as pd
import numpy as np
from typing import List, TypeVar

def calculate_median_prediction_and_label(df):
    """
    This function gets rid of noisy data labels by simply taking the median across unique combinations of experimental setup (disregarding species), i.e.,
    - Endpoint
    - Effect
    - Duration
    - SMILES

    It also counts the number of experiments that has been grouped.

    Outputs:
    pd.Dataframe 
    """
    aggregate_columns = [
        'duration',
        'effect',
        'endpoint', 
        'SMILES_Canonical_RDKit',
        'chemical_name'
        ]
    
    df = df.copy()
    # Get rid of data noise
    df['tmp_id'] = list(range(len(df)))
    medians = df.groupby(aggregate_columns, as_index=False, dropna=False).median(numeric_only=True)
    # Count number of occurences for experimental setup combinations
    counts = df.groupby(aggregate_columns, as_index=False, dropna=False).count()
    
    counts.rename(columns={'tmp_id': 'counts'}, inplace=True)

    medians['counts'] = counts['counts']
    medians.drop(columns=['tmp_id'], inplace=True)
    medians.sort_values(by=['SMILES_Canonical_RDKit'], inplace=True)
    return medians

def calculate_weighted_avg(df):
    """
    This function calculates the weighted average for unique chemical structures, i.e.,
    - Endpoint
    - Effect
    - SMILES

    It does so in the following steps:
    1. Use counts for unique experimental setups (output from CalculateMedianPredictionAndLabel) to weigh each unique experimental setup
    2. Calculate sum across unique chemical, effect, and endpoint combinations (this will now be a weighted sum)
    3. Calculate weighted average by dividing each the sum of counts for that chemical

    Outputs:
    pd.Dataframe 
    """
    df = df.copy()
    for col in df.columns:
        if col not in (['effect','endpoint', 'SMILES_Canonical_RDKit','chemical_name','counts']):
            # Weigh each chemical label and prediction by experiment count
            df[[col]] = df[[col]]*df[['counts']].to_numpy()

    # Sum for each unique chemical (this is a weighted sum)
    mean = df.groupby((['endpoint','SMILES_Canonical_RDKit','chemical_name']), as_index=False, dropna=False).sum(min_count=1, numeric_only=True)

    # Calculate weighted average from the sum by deviding by the sum of summed counts 
    for col in mean.columns:
        try:
            if col not in ['endpoint','SMILES_Canonical_RDKit','chemical_name','counts']:
                mean[col] = mean[[col]]/mean[['counts']].to_numpy()
        except:
            pass
        
    # Calculate residual
    mean.sort_values(by=['SMILES_Canonical_RDKit'], inplace=True)
    return mean