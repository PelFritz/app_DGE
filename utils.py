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
    counts = counts.T
    sample_id = counts.index.tolist()
    scaler = StandardScaler()
    x_std = scaler.fit_transform(counts)
    pca = PCA(n_components=6)
    pcs = pca.fit_transform(x_std)
    pcs = pd.DataFrame(pcs, columns=[f'PC{x}' for x in range(1, pcs.shape[1]+1)])
    pcs['sample_id'] = sample_id
    pcs = pcs.merge(meta, on='sample_id', how='inner')
    return pcs


@st.cache_data
def perform_deg(counts_data_df, meta_data_df, design_factor, contrast, analysis='SingleFactor'):
    # Filter for missing data
    counts_data_df = counts_data_df.T
    counts_data_df.reset_index(names='sample_id', inplace=True)
    data_full = meta_data_df.merge(counts_data_df, on='sample_id', how='inner')
    data_full.dropna(how='any', inplace=True)
    counts_data_df = data_full.copy().drop(columns=[x for x in meta_data_df.columns.tolist() if x != 'sample_id'])
    counts_data_df.set_index('sample_id', inplace=True)
    meta_data_df = data_full[meta_data_df.columns.tolist()]
    meta_data_df.set_index('sample_id', inplace=True)

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
        stat_res = DeseqStats(dds, contrast=contrast, inference=inference, quiet=True, cooks_filter=True, alpha=0.05)
        stat_res.summary()
        results_df = stat_res.results_df
    else:

        stat_res = DeseqStats(dds, contrast=contrast, inference=inference, quiet=True, cooks_filter=True, alpha=0.05,)
        stat_res.summary()
        results_df = stat_res.results_df
    results_df['-log10(pvalue)'] = -np.log10(results_df['pvalue'])
    # Check this
    results_df['regulated'] = ['up' if x > 2 else 'down' if x < -2 else 'stable' for x in results_df['log2FoldChange']]
    results_df['abslog2fc'] = np.abs(results_df['log2FoldChange'])
    results_df = results_df.sort_values(by=['abslog2fc', 'padj'], ascending=[False, True])
    results_df.drop(columns='abslog2fc', inplace=True)
    return results_df, dds


@st.cache_data
def load_data(counts_input, meta_input):
    counts_df = pd.read_csv(counts_input, index_col=0)
    counts_df = counts_df.round().astype('int32')
    meta_df = pd.read_csv(meta_input)
    return counts_df, meta_df
