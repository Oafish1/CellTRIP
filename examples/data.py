import os

import numpy as np
import pandas as pd


def load_data(dataset_name, data_folder):
    spatial_dataset_name_dict = {
        'MouseVisual': 'mouseVisual',
        'ISS': 'ISS',
    }

    if dataset_name in spatial_dataset_name_dict:
        dataset_dir = os.path.join(data_folder, spatial_dataset_name_dict[dataset_name])
        M1 = pd.read_csv(os.path.join(dataset_dir, 's3_cell_by_gene.csv'), delimiter=',', header=1).rename(columns={'gene_name': 'sample_name'})
        # spot_data = pd.read_csv(os.path.join(dataset_dir, 's3_spot_table.csv'), delimiter=',', header=0).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
        M2 = pd.read_csv(os.path.join(dataset_dir, 's3_mapped_cell_table.csv'), delimiter=',', header=0).drop(columns=['Unnamed: 0'])

        # Check all samples line up
        assert (M1['sample_name'] == M2['sample_name']).all()

        # Get final formatted data
        M1 = M1.drop(columns='sample_name')
        F1 = M1.columns
        F1.to_numpy()
        M1 = M1.to_numpy()

        T1 = T2 = M2['layer'].to_numpy()
        F2 = np.array(['x', 'y'])
        M2 = M2[F2].to_numpy()

        modalities = [M1, M2]
        types = [T1, T2]
        features = [F1, F2]

    elif dataset_name == 'scNMT':
        dataset_dir = os.path.join(data_folder, 'UnionCom/scNMT')
        M1 = pd.read_csv(os.path.join(dataset_dir, 'Paccessibility_300.txt'), delimiter=' ', header=None).to_numpy()
        M2 = pd.read_csv(os.path.join(dataset_dir, 'Pmethylation_300.txt'), delimiter=' ', header=None).to_numpy()
        M3 = pd.read_csv(os.path.join(dataset_dir, 'RNA_300.txt'), delimiter=' ', header=None).to_numpy()
        T1 = pd.read_csv(os.path.join(dataset_dir, 'type1.txt'), delimiter=' ', header=None).to_numpy().flatten()
        T2 = pd.read_csv(os.path.join(dataset_dir, 'type2.txt'), delimiter=' ', header=None).to_numpy().flatten()
        T3 = pd.read_csv(os.path.join(dataset_dir, 'type3.txt'), delimiter=' ', header=None).to_numpy().flatten()

        modalities = [M1, M2, M3][1]
        types = [T1, T2, T3][1]
        features = [[i for i in range(M.shape[1])] for M in modalities]

    elif dataset_name == 'BrainChromatin':
        nrows = None  # 2_000
        dataset_dir = os.path.join(data_folder, 'brainchromatin')
        M1 = pd.read_csv(os.path.join(dataset_dir, 'multiome_rna_counts.tsv'), delimiter='\t', nrows=nrows).transpose()  # 4.6 Gb in memory
        M2 = pd.read_csv(os.path.join(dataset_dir, 'multiome_atac_gene_activities.tsv'), delimiter='\t', nrows=nrows).transpose()  # 2.6 Gb in memory
        M2 = M2.transpose()[M1.index].transpose()
        meta = pd.read_csv(os.path.join(dataset_dir, 'multiome_cell_metadata.txt'), delimiter='\t')
        meta_names = pd.read_csv(os.path.join(dataset_dir, 'multiome_cluster_names.txt'), delimiter='\t')
        meta_names = meta_names[meta_names['Assay'] == 'Multiome ATAC']
        meta = pd.merge(meta, meta_names, left_on='ATAC_cluster', right_on='Cluster.ID', how='left')
        meta.index = meta['Cell.ID']
        T1 = T2 = np.array(meta.transpose()[M1.index].transpose()['Cluster.Name'])
        F1, F2 = M1.columns, M2.columns
        M1, M2 = M1.to_numpy(), M2.to_numpy()

        modalities = [M1, M2]
        types = [T1, T2]
        features = [F1, F2]

        del meta, meta_names

    elif dataset_name == 'scGLUE':
        import scanpy as sc
        dataset_dir = os.path.join(data_folder, 'scglue')
        D1 = sc.read_h5ad(os.path.join(dataset_dir, 'Chen-2019-RNA.h5ad'))
        D2 = sc.read_h5ad(os.path.join(dataset_dir, 'Chen-2019-ATAC.h5ad'))
        M1 = np.asarray(D1.X.todense())
        M2 = np.asarray(D2.X.todense())
        T1 = D1.obs.cell_type.to_numpy()
        T2 = D2.obs.cell_type.to_numpy()
        F1 = D1.var.index.to_numpy()
        F2 = D2.var.index.to_numpy()

        modalities = [M1, M2]
        types = [T1, T2]
        features = [F1, F2]

        del D1, D2

    elif dataset_name == 'scGEM':
        dataset_dir = os.path.join(data_folder, 'UnionCom/scGEM')
        M1 = pd.read_csv(os.path.join(dataset_dir, 'GeneExpression.txt'), delimiter=' ', header=None).to_numpy()
        M2 = pd.read_csv(os.path.join(dataset_dir, 'DNAmethylation.txt'), delimiter=' ', header=None).to_numpy()
        T1 = pd.read_csv(os.path.join(dataset_dir, 'type1.txt'), delimiter=' ', header=None).to_numpy()
        T2 = pd.read_csv(os.path.join(dataset_dir, 'type2.txt'), delimiter=' ', header=None).to_numpy()
        F1 = np.loadtxt(os.path.join(dataset_dir, 'gex_names.txt'), dtype='str')
        F2 = np.loadtxt(os.path.join(dataset_dir, 'dm_names.txt'), dtype='str')

        modalities = [M1, M2]
        types = [T1, T2]
        features = [F1, F2]

    # MMD-MA data
    elif dataset_name == 'MMD-MA':
        dataset_dir = os.path.join(data_folder, 'UnionCom/MMD')
        M1 = pd.read_csv(os.path.join(dataset_dir, 's1_mapped1.txt'), delimiter='\t', header=None).to_numpy()
        M2 = pd.read_csv(os.path.join(dataset_dir, 's1_mapped2.txt'), delimiter='\t', header=None).to_numpy()
        T1 = pd.read_csv(os.path.join(dataset_dir, 's1_type1.txt'), delimiter='\t', header=None).to_numpy()
        T2 = pd.read_csv(os.path.join(dataset_dir, 's1_type2.txt'), delimiter='\t', header=None).to_numpy()

        modalities = [M1, M2]
        types = [T1, T2]
        features = [[i for i in range(M.shape[1])] for M in modalities]

    # Random data
    elif dataset_name == 'Random':
        num_nodes = 100
        M1 = np.random.rand(num_nodes, 8)
        M2 = np.random.rand(num_nodes, 16)

        modalities = [M1, M2]
        types = [2*[0 for _ in range(num_nodes)]]
        features = [[i for i in range(M.shape[1])] for M in modalities]

    else: assert False, 'No matching dataset found.'

    return modalities, types, features