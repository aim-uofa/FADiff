# Floating Anchor Diffusion Model for Multi-motif Scaffolding

This repository contains the source code accompanying the paper:

[Floating Anchor Diffusion Model for Multi-motif Scaffolding](https://ai4mol.github.io/projects/FADiff/),  ICML 2024.
 
If you use our work then please cite
```
@article{liu2024floating,
    title={Floating Anchor Diffusion Model for Multi-motif Scaffolding},
    author={Liu, Ke and Mao, Weian and Shen, Shuaike and Jiao, Xiaoran and Sun, Zheng and Chen, Hao and Shen, Chunhua},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=CtgJUQxmEo}
}
```

# Installation

We recommend [miniconda](https://docs.conda.io/en/main/miniconda.html) (or anaconda).
Run the following to install a conda environment with the necessary dependencies.
```bash
conda env create -f FADiff.yml
```

Next, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```

# Training

### Downloading the PDB for training
To get the training dataset, first download PDB then preprocess it with our provided scripts.
PDB can be downloaded from RCSB: https://www.wwpdb.org/ftp/pdb-ftp-sites#rcsbpdb.
Our scripts assume you download in **mmCIF format**.
Navigate down to "Download Protocols" and follow the instructions depending on your location.

> WARNING: Downloading PDB can take up to 1TB of space.

After downloading, you should have a directory formatted like this:
https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/ 
```
00/
01/
02/
..
zz/
```
In this directory, unzip all the files: 
```
gzip -d **/*.gz
```
Then run the following with <path_pdb_dir> replaced with the location of PDB.
```python
python process_pdb_dataset.py --mmcif_dir <pdb_dir> 
```
See the script for more options. Each mmCIF will be written as a pickle file that
we read and process in the data loading pipeline. A `metadata.csv` will be saved
that contains the pickle path of each example as well as additional information
about each example for faster filtering.

For PDB files, we provide some starter code in `process_pdb_files.py`  of how to
modify `process_pdb_dataset.py` to work with PDB files (as we did at an earlier
point in the project). **This has not been tested.** Please make a pull request
if you create a PDB file processing script.

### Downloading PDB clusters
To use clustered training data, download the clusters at 30% sequence identity
at [rcsb](https://www.rcsb.org/docs/programmatic-access/file-download-services#sequence-clusters-data).
This download link also works at time of writing:
```
https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt
```
Place this file in `data/processed_pdb` or anywhere in your file system.
Update your config to point to the clustered data:
```yaml
data:
  cluster_path: ./data/processed_pdb/clusters-by-entity-30.txt
```
To use clustered data, set `sample_mode` to either `cluster_time_batch` or `cluster_length_batch`.
See next section for details.

### Batching modes

```yaml
experiment:
  # Use one of the following.

  # Each batch contains multiple time steps of the same protein.
  sample_mode: time_batch

  # Each batch contains multiple proteins of the same length.
  sample_mode: length_batch

  # Each batch contains multiple time steps of a protein from a cluster.
  sample_mode: cluster_time_batch

  # Each batch contains multiple clusters of the same length.
  sample_mode: cluster_length_batch
```

### Launching training 
```shell
bash train.sh
```
which contains
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=6003
export NCCL_P2P_DISABLE=1
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node=8 \
    --master_port=29504 \
    experiments/train_se3_diffusion.py \
    --config-name=train
```

## License
For non-commercial academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).



## Acknowledgements
This code is built upon FrameDiff "SE(3) diffusion model with application to protein backbone generation": https://arxiv.org/abs/2302.02277
