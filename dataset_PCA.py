import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA

from torch_geometric.data import InMemoryDataset

from pymatgen.core import Structure

from utilities import structureToGraph, getPymatgenElementProperties
from Betti_number import getElementalBettiFeatures

from pymatgen.analysis.defects.finder import DefectSiteFinder

def getDFTEnergy(DFT_dir):
    assert 'data.csv' in os.listdir(DFT_dir), f'data.csv file does not exist in {DFT_dir}.'
    df = pd.read_csv(f'{DFT_dir}/data.csv')
    return df

def getAtomicFeatures(root, atomic_feature_type):
    if 'number' in atomic_feature_type:
        atom_dict = {k: [k] for k in range(1, 100)}
    elif 'pymatgen' in atomic_feature_type:
        atom_dict = {k: getPymatgenElementProperties(k) for k in range(1, 100)}
    elif 'CGCNN' in atomic_feature_type:
        assert 'atomic_embedding_CGCNN.json' in os.listdir(root), f'cannot read atomic_embedding_CGCNN.json file from {root}'
        with open(f'{root}/atomic_embedding_CGCNN.json', 'r') as f:
            embedding = json.load(f)
        atom_dict = {int(key): value for key, value in embedding.items()}
    return atom_dict

def getDistanceToVacancy(pris_struct, defect_struct):
    finder = DefectSiteFinder()
    vac_pos = finder.get_defect_fpos(defect_structure=defect_struct, base_structure=pris_struct)
    vac_pos -= np.round(vac_pos)
    return defect_struct.lattice.get_all_distances(vac_pos, defect_struct.frac_coords)

class Dataset(InMemoryDataset):
    def __init__(self, root, r_cutoff_ripser=None, atomic_feature_type='CGCNN', PCA_components=1, dr=0.1,
                 transform=None, pre_transform=None):
        if not os.path.exists(root):
            os.makedirs(root)

        self.dr = dr
        self.r_cutoff_ripser = r_cutoff_ripser
        self.atomic_feature_type = atomic_feature_type
        self.PCA_components = PCA_components

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    def raw_file_names(self):
        return []

    def processed_file_names(self):
        return ['dataset.pt']

    def download(self): pass

    def process(self):
        if self.r_cutoff_ripser is None:
            raise ValueError('Need to provide r_cutoff_ripser to create the dataset')

        DFT_dir = './datasets/raw'
        df = getDFTEnergy(DFT_dir)
        atom_dict = getAtomicFeatures(root=DFT_dir, atomic_feature_type=self.atomic_feature_type)

        topo_features = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            pris_idx, vac_idx = row['pris_idx'], row['vac_idx']
            if self.r_cutoff_ripser == 0:
                topo_features = None
            else:
                topo_features.append(getElementalBettiFeatures(data_dir=f'{DFT_dir}/Betti_number/{self.r_cutoff_ripser}',
                                                         id=f'{pris_idx}_{vac_idx}', r_cutoff=self.r_cutoff_ripser).numpy())
        topo_features = np.vstack(topo_features)
        pca_transformer = PCA(n_components=self.PCA_components)
        pca_transformer.fit(topo_features)

        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            formation_energy = row['formation_energy']
            pris_idx, vac_idx = row['pris_idx'], row['vac_idx']
            pris_structure = Structure.from_file(f'{DFT_dir}/pristine_structures/{pris_idx}.vasp')
            def_structure = Structure.from_file(f'{DFT_dir}/defective_structures/{pris_idx}_{vac_idx}.vasp')

            dist_to_vac = getDistanceToVacancy(pris_struct=pris_structure, defect_struct=def_structure)

            if self.r_cutoff_ripser == 0:
                topo_features = None
            else:
                topo_features = getElementalBettiFeatures(data_dir=f'{DFT_dir}/Betti_number/{self.r_cutoff_ripser}',
                                                         id=f'{pris_idx}_{vac_idx}', r_cutoff=self.r_cutoff_ripser)
                topo_features = pca_transformer.transform(topo_features)

            data = structureToGraph(structure=def_structure, E=formation_energy, topo_features=topo_features,
                                    atom_dict=atom_dict, dist_to_vac=dist_to_vac, dr=self.dr)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

if __name__ == '__main__':
    r_cutoff_ripser = 10
    atomic_feature_type = 'CGCNN'

    for PCA_components in [1, 2, 4, 6, 8, 10]:
        root = f'datasets/O_vacancies/{r_cutoff_ripser}/{atomic_feature_type}/{PCA_components}'
        dataset = Dataset(root=root, r_cutoff_ripser=r_cutoff_ripser, atomic_feature_type=atomic_feature_type, PCA_components=PCA_components)
