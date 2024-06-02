"""Script for running inference and sampling.

Sample command:
> python scripts/run_inference.py

"""

import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import pandas as pd
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional

from analysis import utils as au
from analysis import metrics
from data import utils as du
from data import residue_constants
from data import all_atom
from typing import Dict
from experiments import train_se3_diffusion
from omegaconf import DictConfig, OmegaConf
from openfold.data import data_transforms
import esm
from data import pdb_data_loader
from openfold.utils import rigid_utils as ru


CA_IDX = residue_constants.atom_order['CA']


def process_chain(design_pdb_feats):
    chain_feats = {
        'aatype': torch.tensor(design_pdb_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(design_pdb_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(design_pdb_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    seq_idx = design_pdb_feats['residue_index'] - np.min(design_pdb_feats['residue_index']) + 1
    chain_feats['seq_idx'] = seq_idx
    chain_feats['res_mask'] = design_pdb_feats['bb_mask']
    chain_feats['residue_index'] = design_pdb_feats['residue_index']
    return chain_feats


def create_pad_feats(pad_amt):
    return {
        'res_mask': torch.ones(pad_amt),
        'fixed_mask': torch.zeros(pad_amt),
        'rigids_impute': torch.zeros((pad_amt, 4, 4)),
        'torsion_impute': torch.zeros((pad_amt, 7, 2)),
    }


class Sampler:

    def __init__(
            self,
            conf: DictConfig,
            conf_overrides: Dict=None
        ):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
        """
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._diff_conf = self._infer_conf.diffusion
        self._sample_conf = self._infer_conf.samples
        self._data_conf = conf.data

        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._infer_conf.pt_hub_dir)

        # Set-up accelerator
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')

        # Set-up directories
        self._weights_path = self._infer_conf.weights_path
        output_dir =self._infer_conf.output_dir
        if self._infer_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        self._output_dir = os.path.join(output_dir, dt_string)
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')
        self._pmpnn_dir = self._infer_conf.pmpnn_dir

        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')

        # Load models and experiment
        self._load_ckpt(conf_overrides)
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)
        
        self._inference_type = self._infer_conf.type
        self._with_sc = self._infer_conf.self_consistency
        if 'motif' in self._inference_type:
            self._motif_conf = self._data_conf.motif
            self._motif_dir = self._data_conf.motif_dir
            self.num_sample_motif = self._sample_conf.num_sample_motif
        if self._inference_type == 'motif':
            self.csv = pd.read_csv(os.path.join(self._motif_dir, 'metadata.csv'))
            self.num_sample_motif = self._sample_conf.num_sample_motif
            self.selected_motifs = self.csv.sample(self.num_sample_motif)
        elif self._inference_type == 'rfdiffusion_motif':
            self.selected_motifs = pdb_data_loader.MotifPdbDataset(
                data_conf=self._data_conf,
                motif_type=self._inference_type,
            )
        elif self._inference_type == 'sz_motif':
            self.selected_motifs = pdb_data_loader.MotifPdbDataset(
                data_conf=self._data_conf,
                motif_type=self._inference_type,
            )
        elif self._inference_type == 'double_sz_motif':
            self.selected_motifs = pdb_data_loader.MotifPdbDataset(
                data_conf=self._data_conf,
                motif_type=self._inference_type,
                num_motif=2,
            )
        elif self._inference_type == 'triple_sz_motif':
            self.selected_motifs = pdb_data_loader.MotifPdbDataset(
                data_conf=self._data_conf,
                motif_type=self._inference_type,
                num_motif=3,
            )
        elif self._inference_type == 'quad_sz_motif':
            self.selected_motifs = pdb_data_loader.MotifPdbDataset(
                data_conf=self._data_conf,
                motif_type=self._inference_type,
                num_motif=4,
            )
        elif self._inference_type == 'five_sz_motif':
            self.selected_motifs = pdb_data_loader.MotifPdbDataset(
                data_conf=self._data_conf,
                motif_type=self._inference_type,
                num_motif=5,
            )


    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f'Loading weights from {self._weights_path}')

        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(
            self._weights_path, use_torch=True,
            map_location=self.device)

        # Merge base experiment config with checkpoint config.
        self._conf.model = OmegaConf.merge(
            self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_se3_diffusion.Experiment(
            conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {
            k.replace('module.', ''):v for k,v in model_weights.items()}
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

    def init_data(
            self,
            *,
            rigids_impute,
            torsion_impute,
            fixed_mask,
            res_mask,
        ):
        num_res = res_mask.shape[0]
        diffuse_mask = (1 - fixed_mask) * res_mask
        fixed_mask = fixed_mask * res_mask

        ref_sample = self.diffuser.sample_ref(
            n_samples=num_res,
            rigids_impute=rigids_impute,
            diffuse_mask=diffuse_mask,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, num_res+1)
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx * res_mask,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': torsion_impute,
            'sc_ca_t': torch.zeros_like(rigids_impute.get_trans()),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)
        return init_feats

    def run_sampling(self):
        if 'motif' in self._inference_type:
            self.run_sampling_motif()
        else:
            self.run_sampling_uncondition()

    def run_sampling_motif(self):
        """Sets up inference run.

        All outputs are written to 
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """
        all_sample_lengths = range(
            self._sample_conf.min_length,
            self._sample_conf.max_length+1,
            self._sample_conf.length_step
        )

        for idx in range(len(self.selected_motifs)):
            if self._inference_type == "motif":
                csv_row = self.selected_motifs.iloc[idx]
                pdb_id = csv_row['pdb_id']
                motif_dict_filename = csv_row['motif_dict_filename'].split('/')[-1]
                motif_dict = torch.load(os.path.join(self._motif_dir, motif_dict_filename))
            elif self._inference_type == 'rfdiffusion_motif':
                motif_dict, pdb_id = self.selected_motifs[idx]
            elif 'sz_motif' in self._inference_type:
                motif_dict, pdb_id = self.selected_motifs[idx]
            else:
                raise NotImplementedError('inference type not implemented')            
                
            motif_output_dir = os.path.join(self._output_dir, f'{pdb_id}')
            os.makedirs(motif_output_dir, exist_ok=True)

            # TODO: save motif to pdb
            # move motif to_the_center 
            motif_tensor_7 = motif_dict['motif_rigids_0']
            motif_rigids = ru.Rigid.from_tensor_7(motif_tensor_7)
            motif_trans = motif_rigids.get_trans()
            motif_rots = motif_rigids.get_rots()
            # 1. gap between motifs
            num_motif, seq_len, xyz_dim = motif_trans.shape
            xyz_max = torch.tensor(np.array([0,0,0]))
            for i in range(num_motif):
                xyz_min, _ = torch.min(motif_trans[i][motif_dict['motif_masks'][i]],dim=0)
                motif_trans[i][motif_dict['motif_masks'][i]] -= xyz_min
                motif_trans[i][motif_dict['motif_masks'][i],0] += xyz_max[0]
                xyz_max, _ = torch.max(motif_trans[i][motif_dict['motif_masks'][i]],dim=0)
                xyz_min_now, _ = torch.min(motif_trans[i][motif_dict['motif_masks'][i]],dim=0)
                xyz_max[0] += 5

            # 2. move them to the center
            center = motif_trans[motif_dict['motif_masks']].mean(0)
            motif_trans[motif_dict['motif_masks']] -= center
            motif_dict['motif_rigids_0'] = ru.Rigid(motif_rots, motif_trans).to_tensor_7()
            #scale = torch.sqrt(max(torch.sqrt((motif_trans[motif_dict['motif_masks']]**2).sum(-1)))).item()
            scale = max(torch.sqrt((motif_trans[motif_dict['motif_masks']]**2).sum(-1))).item()

            atom37_t = all_atom.compute_backbone(
                ru.Rigid.from_tensor_7(motif_dict['motif_rigids_0']), motif_dict['motif_torsion_angles_sin_cos'][...,2,:])[0]
            atom37_t = du.move_to_np(atom37_t)
            for idx in range(motif_dict['motif_aatype'].shape[0]):
                motif_pdb_path = os.path.join(motif_output_dir, f'motif_{idx}.pdb')
                # save atom37 to pdb
                diffuse_mask  = np.ones(motif_dict['motif_masks'][motif_dict['motif_masks']].shape[0])
                b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))
                motif_pdb_path = au.write_prot_to_pdb(
                    atom37_t[idx][motif_dict['motif_masks'][idx]][None,...],
                    motif_pdb_path,
                    b_factors=b_factors,
                    aatype=motif_dict['motif_aatype'][idx][motif_dict['motif_masks'][idx]].numpy(),
                )
            sc_results = []
            for sample_length in all_sample_lengths:
                num_points = sample_length - motif_dict['motif_masks'].sum()
                #scale *= np.exp((float(num_points)/float(sample_length)))
                #scale = 1.5*scale + 1
                scale = 4
                length_dir = os.path.join(
                    self._output_dir, f'{pdb_id}', f'length_{sample_length}')
                os.makedirs(length_dir, exist_ok=True)
                self._log.info(f'Sampling length {sample_length}: {length_dir}')
                for sample_i in range(self._sample_conf.samples_per_length):
                    sample_dir = os.path.join(length_dir, f'sample_{sample_i}')
                    if os.path.isdir(sample_dir):
                        continue
                    os.makedirs(sample_dir, exist_ok=True)
                    sample_output = self.sample(sample_length, motif_dict=motif_dict, scale=scale)
                    traj_paths = self.save_traj(
                        sample_output['prot_traj'],
                        sample_output['rigid_0_traj'],
                        np.ones(sample_length),
                        output_dir=sample_dir,
                        aatype=sample_output['aatype']
                    )
                    fix_pos = np.where(sample_output['motif_position'])[0]
                    fix_pos_list = [str(x) for x in fix_pos]
                    #fix_pos_str = np.array2string(fix_pos, separator=' ')
                    #fix_pos_str_clean = fix_pos_str.strip('[').strip(']')
                    fix_pos_str_clean = " ".join(fix_pos_list)
                    pdb_path = traj_paths['sample_path']
                    if self._with_sc:
                        # Run ProteinMPNN
                        sc_output_dir = os.path.join(sample_dir, 'self_consistency')
                        os.makedirs(sc_output_dir, exist_ok=True)
                        shutil.copy(pdb_path, os.path.join(
                            sc_output_dir, os.path.basename(pdb_path)))
                        max_sctm = self.run_self_consistency(
                            sc_output_dir,
                            pdb_path,
                            self._motif_dir,
                            motif_mask=None,
                            fix_motif = fix_pos_str_clean,
                        )
                        sc_results.append({
                            'pdb_id': pdb_id,
                            'sample_length': sample_length,
                            'sample_idx': sample_i,
                            'max_sctm': max_sctm,
                        })
                    self._log.info(f'Done sample {sample_i}: {pdb_path}')
            sc_csv_path = os.path.join(self._output_dir, f'{pdb_id}', 'sc_results.csv')
            sc_results = pd.DataFrame(sc_results)
            sc_results.to_csv(sc_csv_path)
        
    def run_sampling_uncondition(self):
        """Sets up inference run.

        All outputs are written to 
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """        
        all_sample_lengths = range(
            self._sample_conf.min_length,
            self._sample_conf.max_length+1,
            self._sample_conf.length_step
        )
        for sample_length in all_sample_lengths:
            length_dir = os.path.join(
                self._output_dir, f'length_{sample_length}')
            os.makedirs(length_dir, exist_ok=True)
            self._log.info(f'Sampling length {sample_length}: {length_dir}')
            for sample_i in range(self._sample_conf.samples_per_length):
                sample_dir = os.path.join(length_dir, f'sample_{sample_i}')
                if os.path.isdir(sample_dir):
                    continue
                os.makedirs(sample_dir, exist_ok=True)
                sample_output = self.sample(sample_length)
                traj_paths = self.save_traj(
                    sample_output['prot_traj'],
                    sample_output['rigid_0_traj'],
                    np.ones(sample_length),
                    output_dir=sample_dir
                )

                pdb_path = traj_paths['sample_path']
                if self._with_sc:
                    # Run ProteinMPNN
                    sc_output_dir = os.path.join(sample_dir, 'self_consistency')
                    os.makedirs(sc_output_dir, exist_ok=True)
                    shutil.copy(pdb_path, os.path.join(
                        sc_output_dir, os.path.basename(pdb_path)))
                    _ = self.run_self_consistency(
                        sc_output_dir,
                        pdb_path,
                        motif_mask=None
                    )
                self._log.info(f'Done sample {sample_i}: {pdb_path}')

    def save_traj(
            self,
            bb_prot_traj: np.ndarray,
            x0_traj: np.ndarray,
            diffuse_mask: np.ndarray,
            output_dir: str,
            aatype: np.array,
        ):
        """Writes final sample and reverse diffusion trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file os all intermediate diffused states.
                'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for diffused residues and 0 for motif
            residues if there are any.
        """

        # Write sample.
        diffuse_mask = diffuse_mask.astype(bool)
        sample_path = os.path.join(output_dir, 'sample')
        prot_traj_path = os.path.join(output_dir, 'bb_traj')
        x0_traj_path = os.path.join(output_dir, 'x0_traj')

        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        sample_path = au.write_prot_to_pdb(
            bb_prot_traj[0],
            sample_path,
            b_factors=b_factors,
            aatype=aatype
        )
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            aatype=aatype
        )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            aatype=aatype
        )
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            'x0_traj_path': x0_traj_path,
        }

    def run_self_consistency(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str,
            motif_dir: str,
            fix_motif: str,
            motif_mask: Optional[np.ndarray]=None):
        """Run self-consistency on design proteins against reference protein.

        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            #f'{self._pmpnn_dir}/helper_scripts/reorder_res.py',
            f'{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            #f'--motif_length_file={motif_dir}',
            f'--input_path={decoy_pdb_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        ### 
        path_for_assigned_chains = os.path.join(decoy_pdb_dir, "assigned_pdbs.jsonl")
        chains_to_design = "A"
        process = subprocess.Popen([
            'python',
            #f'{self._pmpnn_dir}/helper_scripts/reorder_res.py',
            f'{self._pmpnn_dir}/helper_scripts/assign_fixed_chains.py',
            #f'--motif_length_file={motif_dir}',
            f'--input_path={output_path}',
            f'--output_path={path_for_assigned_chains}',
            f'--chain_list={chains_to_design}'
        ])
        _ = process.wait()
        ###
        chains_to_design = "A"
        path_for_fixed_positions = os.path.join(decoy_pdb_dir, "fixed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            #f'{self._pmpnn_dir}/helper_scripts/reorder_res.py',
            f'{self._pmpnn_dir}/helper_scripts/make_fixed_positions_dict.py',
            #f'--motif_length_file={motif_dir}',
            f'--input_path={output_path}',
            f'--output_path={path_for_fixed_positions}',
            f'--chain_list={chains_to_design}',
            f'--position_list={fix_motif}'
        ])
        _ = process.wait()
        

        num_tries = 0
        ret = -1
        pmpnn_args = [
            'python',
            f'{self._pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            decoy_pdb_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(self._sample_conf.seq_per_sample),
            '--chain_id_jsonl',
            path_for_assigned_chains,
            '--fixed_positions_jsonl',
            path_for_fixed_positions,
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
        ]
        if self._infer_conf.gpu_id is not None:
            pmpnn_args.append('--device')
            pmpnn_args.append(str(self._infer_conf.gpu_id))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            'seqs',
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            'tm_score': [],
            'sample_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results['motif_rmsd'] = []
        esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['bb_positions'], esmf_feats['bb_positions'])
            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(
                    sample_motif, of_motif)
                mpnn_results['motif_rmsd'].append(motif_rmsd)
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(esmf_sample_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)
        return mpnn_results['tm_score'].max()

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output
    
    def adjust_mask(self, sample_length: int, motif_seq_mask):
        res_mask = np.zeros((motif_seq_mask.shape[0], sample_length), dtype=np.bool_)
        motif_position = np.zeros((sample_length), dtype=np.bool_)
        start_ind = 0
        for i in range(motif_seq_mask.shape[0]):
            length = motif_seq_mask[i,:].sum()
            res_mask[i,start_ind:start_ind+length] = True
            start_ind += length
            motif_position = (res_mask[i] | motif_position)
        return res_mask, motif_position

    def adjust_mask_motif(self, sample_length, motif_dict):
        num_motif, max_motif_len = motif_dict['motif_masks'].shape
        res_mask = np.zeros((num_motif, sample_length), dtype=np.bool_)
        motif_position = np.zeros((sample_length), dtype=np.bool_)
        start_ind = 0
        torsion_angles_sin_cos = np.zeros((sample_length, 7, 2))
        for i in range(num_motif):
            length = motif_dict['motif_masks'][i,:].sum()
            res_mask[i,start_ind:start_ind+length] = True
            motif_position = (res_mask[i] | motif_position)
            torsion_angles_sin_cos[start_ind:start_ind+length] = motif_dict['motif_torsion_angles_sin_cos'][i,motif_dict['motif_masks'][i]]
            start_ind += length
        return torsion_angles_sin_cos, res_mask, motif_position

    def sample(self, sample_length: int, motif_dict=None, scale=None):

        """Sample based on length.

        Args:
            sample_length: length to sample

        Returns:
            Sample outputs. See train_se3_diffusion.inference_fn.
        """
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)
        # Initialize data
        # adjust mask
        if 'sz_motif' in self._inference_type or 'rf' in self._inference_type:
            torsion_angles_sin_cos, motif_dict['motif_seq_masks'], motif_dict['motif_position'] = self.adjust_mask_motif(sample_length, motif_dict)
        else:
            motif_dict['motif_seq_masks'], motif_dict['motif_position'] = self.adjust_mask(sample_length, motif_dict['motif_seq_masks'])
            torsion_angles_sin_cos = np.zeros((sample_length, 7, 2))
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
            motif_dict=motif_dict,
            motif_mask=motif_dict['motif_seq_masks'],
            scale=scale,
        )
        # adjust res_idx
        res_idx = torch.arange(1, sample_length+1)
        '''
        res_idx_copy = copy.deepcopy(res_idx)
        seq_masks = torch.ones(motif_dict['motif_seq_masks'].shape[-1], dtype=torch.bool)
        for i in range(motif_dict['motif_seq_masks'].shape[0]):
            seq_masks *= (~motif_dict['motif_seq_masks'][i,:])
        middle = seq_masks.sum()//2
        idx = 1
        for i in range(len(res_idx)):
            if seq_masks[i]:
                if middle == idx:
                    idx += 1
                res_idx_copy[i] = idx
                idx += 1
            else:
                res_idx_copy[i] = middle
        '''
        # create input feature
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': torsion_angles_sin_cos,
            'sc_ca_t': np.zeros((sample_length, 3)),
            **ref_sample,
        }
        init_psi = init_feats['torsion_angles_sin_cos'][..., 2, :]
        init_atom37, atom37_mask, _, _ = all_atom.compute_backbone(
            ru.Rigid.from_tensor_7(torch.tensor(init_feats['rigids_t'])), torch.tensor(init_psi))

        if 'motif' in self._inference_type and motif_dict is not None:
            init_feats = init_feats | motif_dict
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)

        #dict_keys(['prot_traj', 'rigid_traj', 'trans_traj', 'psi_pred', 'rigid_0_traj'])
        #motif_dict dict_keys(['motif_aatype', 'motif_rigids_0', 'motif_masks', 'motif_seq_masks'])
        num_motif, seq_len = motif_dict['motif_seq_masks'].shape

        ind_res = reorder(init_atom37.numpy(), motif_dict['motif_seq_masks'])
        reorder_feats = reorder_structure(init_feats, ind_res)  
        init_feats.update(reorder_feats)
        for key in ref_sample.keys():
            init_feats[key] =init_feats[key].to(dtype=init_feats['rigids_t'].dtype)
        # Run inference
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t,
            aux_traj=True,
            noise_scale=self._diff_conf.noise_scale
        )
        res = tree.map_structure(lambda x: x[:, 0], sample_out)
        seq_aatype = np.zeros(seq_len, dtype=int)
        for i in range(num_motif):
            index = 0
            for j in range(seq_len):
                if init_feats['motif_seq_masks'][0][i][j]:
                    seq_aatype[j] = int(init_feats['motif_aatype'][0][i][index])
                    index += 1
        res['aatype'] = seq_aatype
        res['motif_position'] = init_feats['motif_position'][0].cpu().numpy()
        return res

import copy
def reorder_structure(input_feat, ind_res):
    #reorder_feats['torsion_angles_sin_cos'].shape  torch.Size([1, 160, 7, 2])
    res = copy.deepcopy(input_feat)
    for i in range(len(ind_res)):
        res['res_mask'][:,i] = input_feat['res_mask'][:,ind_res[i]]
        res['rigids_t'][:,i,:] = input_feat['rigids_t'][:,ind_res[i],:]
        res['motif_seq_masks'][:,:,i] = input_feat['motif_seq_masks'][:,:,ind_res[i]]
        res['motif_position'][:,i] = input_feat['motif_position'][:,ind_res[i]]
        res['torsion_angles_sin_cos'][0, i] = input_feat['torsion_angles_sin_cos'][0, ind_res[i]]
    return res

def tsp(distance_matrix, start):
    n = len(distance_matrix)
    visited = [start]
    path = [start]
    while len(visited) < n:
        min_distance = float('inf')
        next_node = None
        for i in range(n):
            if i not in visited:
                for j in path:
                    if distance_matrix[i][j] < min_distance:
                        min_distance = distance_matrix[i][j]
                        next_node = i
        visited.append(next_node)
        path.append(next_node)
    return path

from scipy.spatial import distance_matrix
def reorder(coords:np.array, motif_position):
    # N, CA, C, O
    num_motif, seq_len = motif_position.shape
    dist_matrix = distance_matrix(coords[:,0,:], coords[:,2,:])

    for i in range(num_motif):
        for j in range(1, seq_len):
            if motif_position[i][j-1] and motif_position[i][j] and dist_matrix[j, j-1]<4:
                dist_matrix[j, :] = 1e9
                dist_matrix[:, j-1] = 1e9
                dist_matrix[j, j-1] = 0.01

    min_values = np.min(dist_matrix, axis=1)
    start = np.argmax(min_values)

    concat_map = tsp(dist_matrix, start)
    return concat_map

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
