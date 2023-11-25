import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import streamlit as st


@st.cache_data
def perform_pca(counts, meta):
    sample_id = counts['sample_id'].tolist()
    counts = counts.drop(columns='sample_id')
    scaler = StandardScaler()
    x_std = scaler.fit_transform(counts)
    pca = PCA(n_components=6)
    pcs = pca.fit_transform(x_std)
    pcs = pd.DataFrame(pcs, columns=[f'PC{x}' for x in range(1, pcs.shape[1]+1)])
    pcs['sample_id'] = sample_id
    pcs = pcs.merge(meta, on='sample_id', how='inner')
    return pcs


@st.cache_data
def perform_deg(counts_data_df, meta_data_df, design_factor, analysis='SingleFactor', levels=''):
    # Filter for missing data
    data_full = meta_data_df.merge(counts_data_df, on='sample_id', how='inner')
    counts_data_df = data_full.copy().drop(columns=meta_data_df.columns.tolist())
    meta_data_df = data_full[meta_data_df.columns.tolist()]
    # Filter out very low expressed genes
    genes_to_keep = counts_data_df.columns[counts_data_df.sum(axis=0) >= 10]
    counts_data_df = counts_data_df[genes_to_keep]

    # Performing analysis based on condition.
    inference = DefaultInference()
    dds = DeseqDataSet(
        counts=counts_data_df,
        metadata=meta_data_df,
        design_factors=design_factor,
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()
    if analysis == 'SingleFactor':
        stat_res = DeseqStats(dds, inference=inference, quiet=True)
        stat_res.summary()
        results_df = stat_res.results_df
    else:
        contrast = [design_factor[-1]]
        contrast.extend(levels)
        stat_res = DeseqStats(dds, contrast=contrast, inference=inference, quiet=True)
        stat_res.summary()
        results_df = stat_res.results_df
    results_df['-log10(padj)'] = -np.log10(results_df['padj'])
    # Check this
    results_df['regulated'] = ['up' if x > 2 else 'down' if x < -2 else 'stable' for x in results_df['log2FoldChange']]
    results_df = results_df.sort_values(by='padj', ascending=True)
    return results_df


@st.cache_data
def load_data(counts_input, meta_input):
    counts_df = pd.read_csv(counts_input)
    meta_df = pd.read_csv(meta_input)
    return counts_df, meta_df
