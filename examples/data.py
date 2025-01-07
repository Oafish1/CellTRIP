import os

import numpy as np
import pandas as pd


def load_data(dataset_name, data_folder):
    spatial_dataset_name_dict = {
        'MouseVisual': 'BARISTASeq',
        'BARISTASeq': 'BARISTASeq',
        'ISS': 'ISS',  # TODO: Mixed species.  Also, make sure samples line up! i.e. same time!
        'MERFISH': 'MERFISH',
        'smFISH': 'smFISH',
        'ExSeq': 'ExSeq',
    }

    if dataset_name == 'TemporalBrain':
        import rds2py

        # Get dir
        dataset_dir = os.path.join(data_folder, 'temporalBrain')

        # Load RNA
        rdata = rds2py.parse_rds(os.path.join(dataset_dir, 'GSE204683_count_matrix.RDS'))
        rdata['attributes']['dimnames'] = rdata['attributes']['Dimnames']
        wrapper = rds2py.generics._dispatcher(rdata)
        M1, F1, C1 = wrapper.matrix.T, *[np.array(sl) for sl in wrapper.dimnames]

        # Load ATAC
        rdata = rds2py.parse_rds(os.path.join(dataset_dir, 'GSE204682_count_matrix.RDS'))
        rdata['attributes']['dimnames'] = rdata['attributes']['Dimnames']
        wrapper = rds2py.generics._dispatcher(rdata)
        M2, F2, C2 = wrapper.matrix.T, *[np.array(sl) for sl in wrapper.dimnames]

        # Assert same order
        assert (C1 == C2).all()

        # Load barcodes
        barcodes1 = pd.read_csv(os.path.join(dataset_dir, 'GSE204683_barcodes.tsv'), delimiter='\t')
        barcodes2 = pd.read_csv(os.path.join(dataset_dir, 'GSE204682_barcodes.tsv'), delimiter='\t')

        # Get sample ids
        uniq_col, count_col = np.unique([e.split('_')[0] for e in C1], return_counts=True)
        assert np.unique(count_col, return_counts=True)[1].max() == 1  # No duplicate counts, otherwise manual annotation is needed
        # Get donor ids
        uniq_donor, count_donor = np.unique(barcodes1['Donor ID'], return_counts=True)
        assert (np.sort(count_col) == np.sort(count_donor)).all()
        # Convert (aided by `preprocessing.R`)
        name_to_id = {d: c for c, d in zip(uniq_col[np.argsort(count_col)], uniq_donor[np.argsort(count_donor)])}

        # Set indices
        barcodes1['Cell ID'] = barcodes1.apply(lambda r: f'{name_to_id[r["Donor ID"]]}_{r["Barcode"]}', axis=1)
        barcodes2['Cell ID'] = barcodes2.apply(lambda r: f'{name_to_id[r["Donor ID"]]}_{r["Barcode"]}', axis=1)
        assert (barcodes1.set_index('Cell ID') == barcodes1.set_index('Cell ID').loc[C1]).all().all()  # For some reason, `barcodes2` doesn't line up with `C2`, so we assume both meta are the correct order

        # Join meta
        barcodes = barcodes1.join(barcodes2, lsuffix=' RNA', rsuffix=' ATAC')
        lsuffix, rsuffix = ' RNA', ' ATAC'
        for col in ('Donor ID', 'Cell type'):
            assert (barcodes[col+lsuffix] == barcodes[col+rsuffix]).all()
            barcodes[col] = barcodes[col+lsuffix]
            barcodes = barcodes.drop(columns=[col+lsuffix, col+rsuffix])

        # Extract cell types
        T1 = T2 = barcodes[['Cell type', 'Donor ID']].to_numpy()

        # TODO: Maybe use meta for actual year/week measurements

        modalities = [M1, M2]
        types = [T1, T2]
        features = [F1, F2]
        meta = barcodes

    elif dataset_name in spatial_dataset_name_dict:
        dataset_dir = os.path.join(data_folder, spatial_dataset_name_dict[dataset_name])

        # Special handling
        if dataset_name not in ('ExSeq',):
            M1 = pd.read_csv(os.path.join(dataset_dir, 's3_cell_by_gene.csv'), delimiter=',', header=1).rename(columns={'gene_name': 'sample_name'})
            spot_data = pd.read_csv(os.path.join(dataset_dir, 's3_spot_table.csv'), delimiter=',', header=0).drop(columns=['Unnamed: 0'])
            M2 = pd.read_csv(os.path.join(dataset_dir, 's3_mapped_cell_table.csv'), delimiter=',', header=0).drop(columns=['Unnamed: 0'])
        else:
            M1 = pd.read_csv(os.path.join(dataset_dir, 's3_cell_by_gene.csv'), delimiter=',', header=0).rename(columns={'Unnamed: 0': 'sample_name'})
            spot_data = pd.read_csv(os.path.join(dataset_dir, 's3_spot_table.csv'), delimiter=',', header=0).drop(columns=['Unnamed: 0'])
            M2 = pd.read_csv(os.path.join(dataset_dir, 's3_mapped_cell_table.csv'), delimiter=',', header=0).drop(columns=['Unnamed: 0']).rename(columns={'cell': 'sample_name', 'cluster': 'layer'})  # Not really a layer, just for consistency

        # Check all samples line up
        assert (M1['sample_name'].reset_index(drop=True) == M2['sample_name'].reset_index(drop=True)).all()

        # Get final formatted data
        M1 = M1.drop(columns='sample_name')
        F1 = M1.columns
        F1 = F1.to_numpy()
        M1 = M1.to_numpy()

        # TODO: Add sample/snapshot
        T1 = T2 = M2[['layer']].to_numpy()
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
        T1 = pd.read_csv(os.path.join(dataset_dir, 'type1.txt'), delimiter=' ', header=None).to_numpy()
        T2 = pd.read_csv(os.path.join(dataset_dir, 'type2.txt'), delimiter=' ', header=None).to_numpy()
        T3 = pd.read_csv(os.path.join(dataset_dir, 'type3.txt'), delimiter=' ', header=None).to_numpy()

        modalities = [M1, M2, M3][1:2]
        types = [T1, T2, T3][1:2]
        features = [np.arange(m.shape[1]) for m in modalities]

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
        T1 = T2 = meta.transpose()[M1.index].transpose()[['Cluster.Name']].to_numpy()
        F1, F2 = M1.columns.to_numpy(), M2.columns.to_numpy()
        M1, M2 = M1.to_numpy(), M2.to_numpy()

        modalities = [M1, M2]
        types = [T1, T2]
        features = [F1, F2]
        meta = meta

        del meta, meta_names

    elif dataset_name == 'scGLUE':
        import scanpy as sc
        dataset_dir = os.path.join(data_folder, 'scglue')
        D1 = sc.read_h5ad(os.path.join(dataset_dir, 'Chen-2019-RNA.h5ad'))
        D2 = sc.read_h5ad(os.path.join(dataset_dir, 'Chen-2019-ATAC.h5ad'))
        M1 = np.asarray(D1.X.todense())
        M2 = np.asarray(D2.X.todense())
        T1 = D1.obs.cell_type.to_numpy().reshape((-1, 1))
        T2 = D2.obs.cell_type.to_numpy().reshape((-1, 1))
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
        features = [np.arange(m.shape[1]) for m in modalities]

    # Random data
    elif dataset_name == 'Random':
        num_nodes = 100
        M1 = np.random.rand(num_nodes, 8)
        M2 = np.random.rand(num_nodes, 16)

        modalities = [M1, M2]
        types = [np.array([0 for _ in range(num_nodes)]).reshape((-1, 1)) for _ in range(len(modalities))]
        features = [np.arange(m.shape[1]) for m in modalities]

    else: assert False, 'No matching dataset found.'

    return modalities, types, features