import os
import pandas as pd
import numpy as np
import pickle as pkl
import subprocess
import argparse
import shutil
from shutil import which
from collections import Counter
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
import  torch.utils.data as Data
from model import Transformer


#############################################################
########################  Parameters  #######################
#############################################################

parser = argparse.ArgumentParser(description="""PhaMer is a python library for identifying bacteriophages from metagenomic data. 
                                 PhaMer is based on a Transorfer model and rely on protein-based vocabulary to convert DNA sequences into sentences.""")
parser.add_argument('--contigs', help='FASTA file of contigs',  default = 'test_contigs.fa')
parser.add_argument('--proteins', help='FASTA file of predicted proteins (optional)')
parser.add_argument('--len', help='minimum length of contigs', type=int, default=300)
parser.add_argument('--threads', help='number of threads to use', type=int, default=8)
parser.add_argument('--dbdir', help='database directory (optional)',  default = 'database')
parser.add_argument('--midfolder', help='folder to store the intermediate files', type=str, default='temp/')
parser.add_argument('--out', help='name of the output file',  type=str, default = 'out/example_prediction.csv')
parser.add_argument('--reject', help='threshold to reject prophage',  type=float, default = 0.3)
parser.add_argument('--query_cover', help='the query-cover value of diamond blastp. 0 indicates no constrain.',  type=int, default = 40)

inputs = parser.parse_args()

#############################################################
######################  Check folders  ######################
#############################################################

out_fn = inputs.midfolder
transformer_fn = inputs.midfolder

if not os.path.isdir(out_fn):
    os.makedirs(out_fn)

db_dir = inputs.dbdir
if not os.path.exists(db_dir):
    print(f'Database directory {db_dir} missing or unreadable')
    exit(1)

out_dir = os.path.dirname(inputs.out)
if out_dir != '':
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


#############################################################
##################  Filter short contigs  ###################
#############################################################
rec = []
for record in SeqIO.parse(inputs.contigs, 'fasta'):
    if len(record.seq) > inputs.len:
        rec.append(record)
SeqIO.write(rec, f'{out_fn}/filtered_contigs.fa', 'fasta')

#############################################################
####################  Prodigal translation  #################
#############################################################

threads = inputs.threads

if inputs.proteins is None:
    prodigal = "prodigal"
    # check if pprodigal is available
    if which("pprodigal") is not None:
        print("Using parallelized prodigal...")
        prodigal = f'pprodigal -T {threads}'

    prodigal_cmd = f'{prodigal} -i {out_fn}/filtered_contigs.fa -a {out_fn}/test_protein.fa -f gff -p meta'
    print("Running prodigal...")
    _ = subprocess.check_call(prodigal_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    if not os.path.exists(f'{out_fn}/test_protein.fa'):
        shutil.copyfile(inputs.proteins, f'{out_fn}/test_protein.fa')

#############################################################
####################  DIAMOND CREATEDB ######################
#############################################################


print("\n\n" + "{:-^80}".format("Diamond BLASTp"))
print("Creating Diamond database and running Diamond...")

diamond_db = f'{db_dir}/database.dmnd'

try:
    if os.path.exists(diamond_db):
        print(f'Using preformatted DIAMOND database ({diamond_db}) ...')
    else:
        # create database
        make_diamond_cmd = f'diamond makedb --threads {threads} --in {db_dir}/database.fa -d {out_fn}/database.dmnd'
        print("Creating Diamond database...")
        _ = subprocess.check_call(make_diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        diamond_db = f'{out_fn}/database.dmnd'
except:
    print("diamond makedb failed")
    exit(1)

#############################################################
####################  DIAMOND BLASTP  #######################
#############################################################

try:
    # running alignment
    query_cover= inputs.query_cover
    if query_cover == 0:
        diamond_cmd = f'diamond blastp --threads {threads} --sensitive -d {diamond_db} -q {out_fn}/test_protein.fa -o {out_fn}/results.tab -k 1 --evalue 1e-5'
    else:
        diamond_cmd = f'diamond blastp --threads {threads} --sensitive -d {diamond_db} -q {out_fn}/test_protein.fa -o {out_fn}/results.tab -k 1 --query-cover {query_cover} --evalue 1e-5'
    print("Running Diamond...")
    _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    diamond_out_fp = f"{out_fn}/results.tab"
    database_abc_fp = f"{out_fn}/results.abc"
    _ = subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
except:
    print("diamond blastp failed")
    exit(1)

#############################################################
####################  Contig2Sentence  ######################
#############################################################



# Load dictonary and BLAST results
proteins_df = pd.read_csv(f'{db_dir}/proteins.csv')
proteins_df.dropna(axis=0, how='any', inplace=True)
pc2wordsid = {pc: idx for idx, pc in enumerate(sorted(set(proteins_df['cluster'].values)))}
protein2pc = {protein: pc for protein, pc in zip(proteins_df['protein_id'].values, proteins_df['cluster'].values)}
blast_df = pd.read_csv(f"{out_fn}/results.abc", sep=' ', names=['query', 'ref', 'evalue'])

# Parse the DIAMOND results
contig2pcs = {}
for query, ref, evalue in zip(blast_df['query'].values, blast_df['ref'].values, blast_df['evalue'].values):
    conitg = query.rsplit('_', 1)[0]
    idx    = query.rsplit('_', 1)[1]
    pc     = pc2wordsid[protein2pc[ref]]
    try:
        contig2pcs[conitg].append((idx, pc, evalue))
    except:
        contig2pcs[conitg] = [(idx, pc, evalue)]

# Sorted by position
for contig in contig2pcs:
    contig2pcs[contig] = sorted(contig2pcs[contig], key=lambda tup: tup[0])



# Contigs2sentence
contig2id = {contig:idx for idx, contig in enumerate(contig2pcs.keys())}
id2contig = {idx:contig for idx, contig in enumerate(contig2pcs.keys())}
sentence = np.zeros((len(contig2id.keys()), 300))
sentence_weight = np.ones((len(contig2id.keys()), 300))
for row in range(sentence.shape[0]):
    contig = id2contig[row]
    pcs = contig2pcs[contig]
    for col in range(len(pcs)):
        try:
            _, sentence[row][col], sentence_weight[row][col] = pcs[col]
            sentence[row][col] += 1
        except:
            break

# Corresponding Evalue weight
#sentence_weight[sentence_weight<1e-200] = 1e-200
#sentence_weight = -np.log(sentence_weight)/200

# propostion
rec = []
for key in blast_df['query'].values:
    name = key.rsplit('_', 1)[0]
    rec.append(name)
counter = Counter(rec)
mapped_num = np.array([counter[item] for item in id2contig.values()])

rec = []
for record in SeqIO.parse(f'{out_fn}/test_protein.fa', 'fasta'):
    name = record.id
    name = name.rsplit('_', 1)[0]
    rec.append(name)
counter = Counter(rec)
total_num = np.array([counter[item] for item in id2contig.values()])
proportion = mapped_num/total_num

# # Store the parameters
# pkl.dump(sentence,        open(f'{transformer_fn}/sentence.feat', 'wb'))
# pkl.dump(id2contig,       open(f'{transformer_fn}/sentence_id2contig.dict', 'wb'))
# pkl.dump(proportion,      open(f'{transformer_fn}/sentence_proportion.feat', 'wb'))
# pkl.dump(pc2wordsid,      open(f'{transformer_fn}/pc2wordsid.dict', 'wb'))


#############################################################
####################     Load Model    ######################
#############################################################
transformer_fn = inputs.midfolder
pcs2idx = pc2wordsid
num_pcs = len(set(pcs2idx.keys()))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("running with cpu")
    torch.set_num_threads(inputs.threads)

src_pad_idx = 0
src_vocab_size = num_pcs+1


def reset_model():
    model = Transformer(
                src_vocab_size, 
                src_pad_idx, 
                device=device, 
                max_length=300, 
                dropout=0.1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_func


def return_batch(train_sentence, label, flag):
    X_train = torch.from_numpy(train_sentence).to(device)
    y_train = torch.from_numpy(label).float().to(device)
    train_dataset = Data.TensorDataset(X_train, y_train)
    training_loader = Data.DataLoader(
        dataset=train_dataset,    
        batch_size=200,
        shuffle=flag,               
        num_workers=0,              
    )
    return training_loader


def return_tensor(var, device):
    return torch.from_numpy(var).to(device)


def reject_prophage(all_pred, weight):
    all_pred = all_pred.detach().cpu().numpy()
    all_pred[weight < inputs.reject] = 0
    return all_pred


# training with short contigs 
model, optimizer, loss_func = reset_model()
try:
    pretrained_dict=torch.load(f'{db_dir}/transformer.pth', map_location=device, weights_only=True)
    model.load_state_dict(pretrained_dict)
except:
    print('cannot find pre-trained model')
    exit()

####################################################################################
##########################   Predit with contigs    ################################
####################################################################################


# sentence   = pkl.load(open(f'{transformer_fn}/sentence.feat', 'rb'))
# id2contig  = pkl.load(open(f'{transformer_fn}/sentence_id2contig.dict', 'rb'))
# proportion = pkl.load(open(f'{transformer_fn}/sentence_proportion.feat', 'rb'))


all_pred = []
all_score = []
with torch.no_grad():
    _ = model.eval()
    for idx in range(0, len(sentence), 500):
        try:
            batch_x = sentence[idx: idx+500]
            weight  = proportion[idx: idx+500]
        except:
            batch_x = sentence[idx:]
            weight  = proportion[idx:]
        batch_x = return_tensor(batch_x, device).long()
        logit = model(batch_x)
        logit = torch.sigmoid(logit.squeeze(1))
        logit = reject_prophage(logit, weight)
        pred  = ['NCLDV' if item > 0.5 else 'non-NCLDV' for item in logit]
        all_pred += pred
        all_score += [float('{:.3f}'.format(i)) for i in logit]


pred_csv = pd.DataFrame({"Contig":id2contig.values(), "Pred":all_pred, "Score":all_score})
pred_csv.to_csv(inputs.out, index = False)

rec = []
ncldv = pred_csv[pred_csv['Pred'] == 'NCLDV']
for record in SeqIO.parse(inputs.contigs, 'fasta'):
    if record.id in ncldv['Contig'].values:
        rec.append(record)
SeqIO.write(rec, f'{out_fn}/output_ncldv_contigs.fa', 'fasta')
