# from biotransformers import BioTransformers
# import ray

import numpy as np
import pandas as pd

df_fixbb2 = pd.read_csv('muformer2-fixbb-results.csv', usecols=["sequence", "loss", "plddt", "pae", "ptm", "rmsd"])
df_bba2 = pd.read_csv('muformer2-bba-results.csv', usecols=["sequence", "loss", "plddt", "pae", "ptm", "rmsd"])
df_fixbb = pd.read_csv('muformer-fixbb-results.csv', usecols=["sequence", "loss", "plddt", "pae", "ptm", "rmsd"])
df_bba = pd.read_csv('muformer-bba-results.csv', usecols=["sequence", "loss", "plddt", "pae", "ptm", "rmsd"])
df_af_only = pd.read_csv('alphafold2-results.csv', usecols=["sequence", "loss", "plddt", "pae", "ptm", "rmsd"])

sequence = [] 

with open('original-sequence.txt', 'r') as file:
    for line in file:
        sequence.append(line)
        
seq = sequence[0][:993]

ray.init()

bio_trans = BioTransformers(backend="esm1b_t33_650M_UR50S", num_gpus=1)
loglikelihood = bio_trans.compute_loglikelihood(sequences, batch_size=2)

def compute_perplexity(loglikelihood_score):
    return np.exp(-loglikelihood_score)

sequences_fixbb = list(df_fixbb['sequence'])
loglikelihood_fixbb = bio_trans.compute_loglikelihood(sequences_fixbb, batch_size=2)
perplexity_fixbb = [compute_perplexity(x) for x in loglikelihood_fixbb]
df_results_fixbb = pd.DataFrame({'loglikelihood': loglikelihood_fixbb, 'perplexity': perplexity_fixbb})
df_results_fixbb.to_csv('muformer-fixbb-loglikelihood.csv')

sequences_fixbb2 = list(df_fixbb2['sequence'])
loglikelihood_fixbb2 = bio_trans.compute_loglikelihood(sequences_fixbb2, batch_size=2)
perplexity_fixbb2 = [compute_perplexity(x) for x in loglikelihood_fixbb2]
df_results_fixbb2 = pd.DataFrame({'loglikelihood': loglikelihood_fixbb2, 'perplexity': perplexity_fixbb2})
df_results_fixbb2.to_csv('muformer2-fixbb-loglikelihood.csv')

sequences_bba = list(df_bba['sequence'])
loglikelihood_bba = bio_trans.compute_loglikelihood(sequences_bba, batch_size=2)
perplexity_bba = [compute_perplexity(x) for x in loglikelihood_bba]
df_results_bba = pd.DataFrame({'loglikelihood': loglikelihood_bba, 'perplexity': perplexity_bba})
df_results_bba.to_csv('muformer-bba-loglikelihood.csv')

sequences_bba2 = list(df_bba2['sequence'])
loglikelihood_bba2 = bio_trans.compute_loglikelihood(sequences_bba2, batch_size=2)
perplexity_bba2 = [compute_perplexity(x) for x in loglikelihood_bba2]
df_results_bba2 = pd.DataFrame({'loglikelihood': loglikelihood_bba2, 'perplexity': perplexity_bba2})
df_results_bba2.to_csv('muformer-bba-loglikelihood.csv')

sequences_af_only = list(df_af_only['sequence'])
loglikelihood_af_only = bio_trans.compute_loglikelihood(sequences_af_only, batch_size=2)
perplexity_af_only = [compute_perplexity(x) for x in loglikelihood_af_only]
df_results_af_only = pd.DataFrame({'loglikelihood': loglikelihood_af_only, 'perplexity': perplexity_af_only})
df_results_af_only.to_csv('alphafold2-loglikelihood.csv')
