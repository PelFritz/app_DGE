import streamlit as st
from pydeseq2.utils import load_example_data
import plotly.express as px
from utils import load_data, perform_pca, perform_deg
st.set_page_config(layout='wide')
st.write('NB: Please this is still under development and improvement!!!')
st.title(':blue[Differential expression analysis Tool]')
st.subheader('Perform differential gene expression without writing code.')
tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Load Data', 'PCA', 'DEG'])


with tab1:
    st.write("""To perform your DEG analysis, we require two datasets.\n
    1. Counts dataset has the raw counts obtained from mapping using a tool like kallisto. NB: rows=sample, columns=genes
    2. Metadata dataset contains information about the experiment design. The rows are the samples and you should have a
    column for every experiment information. E.g a column telling us which sample belongs to which treatment group.
    Please see the example formats below""")
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
    example_counts.reset_index(inplace=True, names='sample_id')
    example_metadata.reset_index(inplace=True, names='sample_id')

    st.write(example_counts.head(4))
    st.write(example_metadata.head(4))

with tab2:
    counts_data = st.sidebar.file_uploader(label="Load counts data in CSV format")
    meta_data = st.sidebar.file_uploader(label="Load a file for metadata in CVS format")
    if counts_data is not None and meta_data is not None:
        counts_df, meta_df = load_data(counts_input=counts_data, meta_input=meta_data)
        st.write('Please have a look at the first 5 rows of your dataset below and verify if they are loaded properly')
        st.write('Here are your counts data')
        st.write(counts_df.head(5))
        st.write('Here are you meta data')
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
        color_scheme = st.sidebar.selectbox(label='select meta column to color plots',
                                            options=[x for x in meta_df.columns if x != 'sample_id'])

        with col1:
            st.plotly_chart(px.scatter(pcs, x='PC1', y='PC2', color=color_scheme,
                                       title='PC1 vs PC2',
                                       color_discrete_sequence=px.colors.qualitative.Dark24),
                            use_container_width=True)
        with col2:
            st.plotly_chart(px.scatter(pcs, x='PC3', y='PC4', color=color_scheme,
                                       title='PC3 vs PC4',
                                       color_discrete_sequence=px.colors.qualitative.Dark24),
                            use_container_width=True)
        with col3:
            st.plotly_chart(px.scatter(pcs, x='PC5', y='PC6', color=color_scheme,
                                       title='PC5 vs PC6',
                                       color_discrete_sequence=px.colors.qualitative.Dark24),
                            use_container_width=True)

with tab4:
    tab_deg1, tab_deg2 = st.tabs(['Results', 'Plots'])
    analysis_type = st.sidebar.selectbox('Analysis', options=['SingleFactor', 'MultiFactor'])
    if analysis_type == 'SingleFactor':
        condition = st.sidebar.selectbox('Choose factor column',
                                         options=[x for x in meta_df.columns if x != 'sample_id'])
        deg_results = perform_deg(counts_data_df=counts_df, meta_data_df=meta_df,
                                  design_factor=condition)
        levels = meta_df[condition].unique()
        with tab_deg1:
            st.write(f'P_values computed based on: factor = {condition}: levels = {levels}')
            st.write(deg_results)
        with tab_deg2:
            fig = px.scatter(deg_results, x='log2FoldChange', y='-log10(padj)', title='Volcano Plot')
            st.plotly_chart(fig)

    else:
        conditions = st.sidebar.multiselect('Choose multiple factor columns',
                                            options=[x for x in meta_df.columns if x != 'sample_id'],
                                            max_selections=2)
        if len(conditions) == 2:
            levels = st.sidebar.multiselect('Choose levels',
                                            options=meta_df[conditions[-1]].unique(),
                                            max_selections=2)
            if len(levels) == 2:
                deg_results = perform_deg(counts_data_df=counts_df, meta_data_df=meta_df,
                                          design_factor=conditions, levels=levels)
                with tab_deg1:
                    st.write(f'P_values computed based on: factor = {conditions[-1]}: levels = {levels}')
                    st.write(deg_results)
                with tab_deg2:
                    fig = px.scatter(deg_results, x='log2FoldChange', y='-log10(padj)', title='Volcano Plot')
                    st.plotly_chart(fig)

### TO DO
# 1. Verify what is being done at the DEG and exactly which counts is expected. Does it need to be normalized or not
# 2. Check the lagging problem