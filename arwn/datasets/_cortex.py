import csv
import logging
import os
import numpy as np
import pandas as pd
from ._download import download
from arwn.definitions import get_root
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

class CortexDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir + 'cortex'
    def fetch(self):
        """Loads cortex dataset."""
        save_path = os.path.abspath(self.dir)
        url = "https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt"
        save_fn = "expression.bin"
        download(url, save_path, save_fn)
        self.data = self._to_csv(os.path.join(save_path, save_fn))
    def _to_csv(self,path_to_file):
        logger.info("Loading Cortex data from {}".format(path_to_file))
        rows = []
        gene_names = []
        with open(path_to_file, "r") as csvfile:
            data_reader = csv.reader(csvfile, delimiter="\t")
            for i, row in enumerate(data_reader):
                if i == 1:
                    precise_clusters = np.asarray(row, dtype=str)[2:]
                if i == 8:
                    clusters = np.asarray(row, dtype=str)[2:]
                if i >= 11:
                    rows.append(row[1:])
                    gene_names.append(row[0])
        cell_types, labels = np.unique(clusters, return_inverse=True)
        _, precise_labels = np.unique(precise_clusters, return_inverse=True)
        data = np.asarray(rows, dtype=np.int).T[1:]
        gene_names = np.asarray(gene_names, dtype=np.str)
        gene_indices = []

        extra_gene_indices = []
        gene_indices = np.concatenate([gene_indices, extra_gene_indices]).astype(np.int32)
        if gene_indices.size == 0:
            gene_indices = slice(None)

        data = data[:, gene_indices]
        gene_names = gene_names[gene_indices]
        data_df = pd.DataFrame(data, columns=gene_names)
        data_df.to_csv(self.dir + '/cortex.csv')
        
    def _read_csv(self):
        self.data = pd.read_csv(self.dir + '/cortex.csv').to_numpy()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
