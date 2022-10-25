from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch

import esm

import requests

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

pdb_id = "7LWV" 

data = requests.get(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}').json()[pdb_id.lower()]
sequence = data[0]['sequence']

def get_bias_from_esm(seq, p=None):
      '''p=None; number of calculation done in parallel (increase if you have more gpu-memory)'''

  # map esm-alphabet to standard-alphabet
  tmp_a2n = {a:n for n,a in enumerate(alphabet.all_toks[4:24])}
  tmp_aa_map = np.array([tmp_a2n[a] for a in "ARNDCQEGHILKMFPSTWYV"])

  x,ln = alphabet.get_batch_converter()([(None,seq)])[-1],len(seq)
  if p is None: p = ln
  with torch.no_grad():
    f = lambda x: model(x)["logits"][:,1:(ln+1),4:24]
    logits = np.zeros((ln,20))
    for n in range(0,ln,p):
      m = min(n+p,ln)
      x_h = torch.tile(torch.clone(x),[m-n,1])
      for i in range(m-n):
        x_h[i,n+i+1] = alphabet.mask_idx
      fx_h = f(x_h.to(device))
      for i in range(m-n):
        logits[n+i] = fx_h[i,n+i].cpu().numpy()
  
    return logits[:,tmp_aa_map]

bias = get_bias_from_esm(sequence)
np.savetxt("esm2-bias.txt", bias)

plt.rcParams["figure.figsize"] = (50,30)
plt.imshow(bias.T)

# clear GPU memory
del model
# gc.collect()
torch.cuda.empty_cache()