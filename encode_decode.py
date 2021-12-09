import numpy as np
import pandas as pd

# global variables
ALL_AA = '$GALMFWKQESPVICYHRNDT'
ALPHABET = {aa: i for i, aa in enumerate(list(ALL_AA))}
TARGET_LEN = 15

def encode_proteins(seqs, save_path='data/'):
    encoded_seqs = np.zeros((len(seqs), TARGET_LEN*len(ALL_AA)))
    for idx, seq in enumerate(seqs): 
        len_seq = len(seq)
        if len_seq < TARGET_LEN:
            new_seq = seq.rjust(TARGET_LEN, '$')
        elif len_seq > TARGET_LEN:
            new_seq = seq[:TARGET_LEN]
        else:
            new_seq = seq
        cur_start = 0
        for a_idx in range(TARGET_LEN):
            encoded_seqs[idx, cur_start + ALPHABET[new_seq[a_idx]]] = 1
            cur_start += len(ALL_AA)
    encoded_seqs = np.array(encoded_seqs)
    np.save(save_path+'encoded_seqs.npy', encoded_seqs)
    return encoded_seqs

def decode_proteins(encoded_seqs, save_path='data/'):
    seqs = []
    for idx, encoded_seq in enumerate(encoded_seqs): 
        cur_start = 0
        seq = ''
        for i in range(TARGET_LEN):
            aa_idx = encoded_seqs[idx, cur_start : cur_start+len(ALL_AA)].tolist().index(1)
            seq += ALL_AA[aa_idx]
            cur_start += len(ALL_AA)
        seqs.append(seq)
    f = open(save_path+'decoded_seqs.txt', 'w')
    for seq in seqs:
        f.write(seq + "\n")
    return seqs