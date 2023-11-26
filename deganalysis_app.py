import numpy as np
import pandas as pd
import streamlit as st
from pydeseq2.utils import load_example_data
import plotly.express as px
from utils import load_data, perform_pca, perform_deg
import seaborn as sns
st.set_page_config(layout='wide')
st.title(':red[Differential expression analysis Toolkit]')
st.subheader('Perform differential gene expression without writing code.')
tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Data', 'PCA', 'DEG'])


with tab1:
    st.write("""\n
    To perform your DEG analysis, we require two datasets.\n
    1. Counts dataset has the raw counts obtained from mapping using a tool like kallisto. NB: rows=genes,
     columns=sample. The first column should be the gene_ids.
    2. Metadata dataset contains information about the experiment design. The rows are the samples and you should have a
    column for every experiment information. E.g a column telling us which sample belongs to which treatment group.
    Please see the example formats below.
    3. Column names should not have underscore "_" in them; e.g "heat_stress" should be "heat stress".
    
    Currently we support:\n
    1. SingleFactor: Experiments with one factor influencing the counts.
    2. MultiFactor: Experiments with more than one factor influencing the gene expression.""")
    example_counts = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    example_metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )
    example_counts = example_counts.T
    example_counts.reset_index(inplace=True, names='gene_id')
    example_metadata.reset_index(inplace=True, names='sample_id')

    st.write(example_counts.head(5))
    st.write(example_metadata.head(5))

with tab2:
    counts_data = st.sidebar.file_uploader(label="Load counts data in CSV format")
    meta_data = st.sidebar.file_uploader(label="Load a file for metadata in CSV format")
    if counts_data is not None and meta_data is not None:
        counts_df, meta_df = load_data(counts_input=counts_data, meta_input=meta_data)
        st.write('Please have a look at the first 5 rows of your dataset below and verify if they are loaded properly.')
        st.write('Here are your counts data.')
        st.write(counts_df.head(5))
        st.write('Here are you meta data.')
        st.write(meta_df.head(5))
    else:
        st.stop()

pcs = perform_pca(counts_df, meta_df)
with tab3:
    tab_pca1, tab_pca2 = st.tabs(['Principal Components', 'Plots'])
    with tab_pca1:
        st.write('PCA analysis performed')
        st.write(pcs)
    with tab_pca2:
        col1, col2, col3 = st.columns(3)
        color_scheme = st.sidebar.selectbox(label='select column to color PCA plots',
                                            options=[x for x in meta_df.columns if x != 'sample_id'])

        with col1:
            st.plotly_chart(px.scatter(pcs, x='PC1', y='PC2', color=color_scheme,
                                       title='PC1 vs PC2',
                                       color_discrete_sequence=px.colors.qualitative.Dark24).update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)),
                            use_container_width=True)
        with col2:
            st.plotly_chart(px.scatter(pcs, x='PC3', y='PC4', color=color_scheme,
                                       title='PC3 vs PC4',
                                       color_discrete_sequence=px.colors.qualitative.Dark24).update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)),
                            use_container_width=True)
        with col3:
            st.plotly_chart(
                px.scatter(pcs, x='PC5', y='PC6', color=color_scheme, title='PC5 vs PC6',
                           color_discrete_sequence=px.colors.qualitative.Dark24).update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)),
                use_container_width=True)

with tab4:
    tab_deg1, tab_deg2, tab_deg3 = st.tabs(['Results', 'Plots', 'Heatmap'])
    analysis_type = st.sidebar.selectbox('Analysis', options=['SingleFactor', 'MultiFactor'])
    if analysis_type == 'SingleFactor':
        condition = st.sidebar.selectbox('Choose factor column',
                                         options=[x for x in meta_df.columns if x != 'sample_id'])
        levels = st.sidebar.multiselect('Choose contrast levels',
                                        options=meta_df[condition].unique(), max_selections=2)
        if condition and len(levels) == 2:
            contrast = [condition]
            contrast.extend(levels)
            deg_results, dds = perform_deg(counts_data_df=counts_df, meta_data_df=meta_df,
                                           design_factor=condition, contrast=contrast)
            with tab_deg1:
                st.write(f"""\n
                P_values computed based on: factor = {condition}: levels = {levels}.\n
                Results are sorted by log2 fold changes, then adjusted p_values.""")
                st.write(deg_results)
            with tab_deg2:
                col1deg, col2deg = st.columns(2)
                with col1deg:
                    fig = px.scatter(deg_results, x='log2FoldChange', y='-log10(pvalue)', title='Volcano Plot',
                                     color='regulated',
                                     color_discrete_sequence=['tomato', 'cornflowerblue', 'grey'])
                    st.plotly_chart(fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                                  xanchor="right", x=1)),
                                    use_container_width=True)
                with col2deg:
                    fig = px.pie(deg_results, values='baseMean', names='regulated',
                                 color_discrete_sequence=['grey', 'cornflowerblue', 'tomato'], hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
            with tab_deg3:
                col_den1, _ = st.columns(2)
                with col_den1:
                    sigs = deg_results[(deg_results.padj < 0.05) & (abs(deg_results.log2FoldChange > 2))]
                    dds.layers['log1p'] = np.log1p(dds.layers['normed_counts'])
                    dds_sigs = dds[:, sigs.index]
                    dds_sigs_df = pd.DataFrame(dds_sigs.layers['log1p'].T,
                                               index=dds_sigs.var_names,
                                               columns=dds_sigs.obs_names)
                    fig = sns.clustermap(dds_sigs_df, z_score=0, cmap='RdYlBu_r', figsize=(6, 6))
                    st.pyplot(fig, use_container_width=True)

    else:
        conditions = st.sidebar.multiselect('Choose multiple factor columns',
                                            options=[x for x in meta_df.columns if x != 'sample_id'],
                                            max_selections=2)
        if len(conditions) == 2:
            levels = st.sidebar.multiselect('Choose levels',
                                            options=meta_df[conditions[-1]].unique(),
                                            max_selections=2)
            if len(levels) == 2:
                contrast = [conditions[-1]]
                contrast.extend(levels)
                deg_results, dds = perform_deg(counts_data_df=counts_df, meta_data_df=meta_df,
                                               design_factor=conditions, contrast=contrast)
                with tab_deg1:
                    st.write(f"""\n
                    P_values computed based on: factor = {conditions[-1]}: levels = {levels}.\n
                    Results are sorted by log2 fold changes, then adjusted p_values.""")
                    st.write(deg_results)
                with tab_deg2:
                    col1deg, col2deg = st.columns(2)
                    with col1deg:
                        fig = px.scatter(deg_results, x='log2FoldChange', y='-log10(pvalue)', title='Volcano Plot',
                                         color='regulated',
                                         color_discrete_sequence=['tomato', 'cornflowerblue', 'grey'])
                        st.plotly_chart(fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                                      xanchor="right", x=1)),
                                        use_container_width=True)
                        with col2deg:
                            fig2 = px.pie(deg_results, values='baseMean', names='regulated',
                                          color_discrete_sequence=['grey', 'cornflowerblue', 'tomato'], hole=0.4,
                                          title='Pie plot')
                            st.plotly_chart(fig2, use_container_width=True)

                with tab_deg3:
                    col_den1, _ = st.columns(2)
                    with col_den1:
                        sigs = deg_results[(deg_results.padj < 0.05) & (abs(deg_results.log2FoldChange > 2))]
                        dds.layers['log1p'] = np.log1p(dds.layers['normed_counts'])
                        dds_sigs = dds[:, sigs.index]
                        st.write(dds_sigs.obs_names)
                        dds_sigs_df = pd.DataFrame(dds_sigs.layers['log1p'].T,
                                                   index=dds_sigs.var_names,
                                                   columns=dds_sigs.obs_names)
                        fig = sns.clustermap(dds_sigs_df, z_score=0, cmap='RdYlBu_r', figsize=(6, 6))
                        st.pyplot(fig, use_container_width=True)


