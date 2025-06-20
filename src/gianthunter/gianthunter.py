#!/usr/bin/env python
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
from gianthunter.model import Transformer



def run(inputs, out_fn, db_dir, MAX_LENGTH=500):
    #############################################################
    ##################  Filter short contigs  ###################
    #############################################################
    rec = []
    contig2length = {}
    for record in SeqIO.parse(inputs.contigs, 'fasta'):
        if len(record.seq) > inputs.len:
            rec.append(record)
        contig2length[record.id] = len(record.seq)

    if len(rec) == 0:
        print("No contigs longer than the specified length found.")
        exit(1)
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
        _ = subprocess.check_call(prodigal_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        if not os.path.exists(f'{out_fn}/test_protein.fa'):
            shutil.copyfile(inputs.proteins, f'{out_fn}/test_protein.fa')

    #############################################################
    ####################  DIAMOND CREATEDB ######################
    #############################################################


    diamond_db = f'{db_dir}/database.dmnd'

    try:
        if os.path.exists(diamond_db):
            print(f'Using preformatted DIAMOND database ({diamond_db}) ...')
        else:
            # create database
            make_diamond_cmd = f'diamond makedb --threads {threads} --in {db_dir}/database.fa -d {out_fn}/database.dmnd'
            print("Creating database...")
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
            diamond_cmd = f'diamond blastp --threads {threads} --sensitive -d {diamond_db} -q {out_fn}/test_protein.fa -o {out_fn}/results.tab -k 1 --evalue 1e-5 --quiet'
        else:
            diamond_cmd = f'diamond blastp --threads {threads} --sensitive -d {diamond_db} -q {out_fn}/test_protein.fa -o {out_fn}/results.tab -k 1 --query-cover {query_cover} --evalue 1e-5 --quiet'
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
    sentence = np.zeros((len(contig2id.keys()), MAX_LENGTH))
    sentence_weight = np.ones((len(contig2id.keys()), MAX_LENGTH))
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



    #############################################################
    ####################     Load Model    ######################
    #############################################################
    pcs2idx = pc2wordsid
    num_pcs = len(set(pcs2idx.keys()))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("running the model on CPU")
        torch.set_num_threads(inputs.threads)

    src_pad_idx = 0
    src_vocab_size = num_pcs+1


    def reset_model():
        model = Transformer(
                    src_vocab_size, 
                    src_pad_idx, 
                    device=device, 
                    max_length=MAX_LENGTH, 
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
        pretrained_dict=torch.load(f'{db_dir}/transformer.pth', map_location=device)
        model.load_state_dict(pretrained_dict)
    except:
        print('cannot find pre-trained model')
        exit()

    ####################################################################################
    ##########################   Predit with contigs    ################################
    ####################################################################################

    marker = pd.read_csv(f'{db_dir}/marker.csv',index_col=0)
    marker['PCID'] = marker['PCID'].apply(eval)
    id2marker = {}
    for gene, row in marker.iterrows():
        id_list = row['PCID']
        for id in id_list:
            id2marker[id] = gene

    all_pred = []
    all_score = []
    all_marker = []
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

            marker = []
            for i, single_pred in enumerate(pred):
                if single_pred == 'NCLDV':
                    single_sentence = batch_x[i]
                    single_marker_gene = set()
                    for id in single_sentence:
                        id = int(id)
                        if id in id2marker:
                            single_marker_gene.add(id2marker[id])
                    marker.append(';'.join(list(single_marker_gene)))
                else:
                    marker.append('')
            all_marker.extend(marker)
            

    all_length = [contig2length[id2contig[idx]] for idx in id2contig]
    pred_csv = pd.DataFrame({"Contig":id2contig.values(), "Length": all_length, "Pred":all_pred, "Score":all_score, "marker":all_marker})
    pred_csv.to_csv(f'{inputs.out}/GiantHunter_prediction.csv', index = False)

    rec = []
    ncldv = pred_csv[pred_csv['Pred'] == 'NCLDV']
    for record in SeqIO.parse(inputs.contigs, 'fasta'):
        if record.id in ncldv['Contig'].values:
            rec.append(record)

    SeqIO.write(rec, f'{inputs.out}/giant_virus_contigs.fa', 'fasta')
    os.system(f'cp {out_fn}/test_protein.fa {inputs.out}/giant_virus_proteins.fa')

    blast_df = pd.read_csv(f"{out_fn}/results.tab", sep='\t', header=None)
    blast_df['contig'] = blast_df[0].apply(lambda x: x.rsplit('_', 1)[0])
    blast_df = blast_df[blast_df['contig'].isin(ncldv['Contig'].values)]
    acc2protein = pkl.load(open(f'{db_dir}/acc2protein.pkl', 'rb'))
    blast_df['protein'] = blast_df[1].apply(lambda x: acc2protein.get(x, "hypothetical protein"))
    blast_df = blast_df[[0, 'protein', 2]]
    blast_df.columns = ['Protein', 'Annotation', 'Confidence']
    # check if the annotation is a hypothetical protein, then replace its confidence with '-'
    hyperthetical_proteins = blast_df[blast_df['Annotation'] == 'hypothetical protein']
    blast_df.loc[hyperthetical_proteins.index, 'Confidence'] = '-'
    blast_df.to_csv(f'{inputs.out}/protein_annotation.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description="""GiantHunter is a python library for identifying NCLDVs from metagenomic data. 
                                    GiantHunter is based on a Transorfer model and relies on protein-based vocabulary to convert DNA sequences into sentences.""")
    parser.add_argument('--contigs', help='FASTA file of contigs',  default = 'test_contigs.fa')
    parser.add_argument('--proteins', help='FASTA file of predicted proteins (optional)')
    parser.add_argument('--len', help='minimum length of contigs', type=int, default=3000)
    parser.add_argument('--threads', help='number of threads to use', type=int, default=int(os.cpu_count()))
    parser.add_argument('--dbdir', help='database directory (optional)',  default = 'database')
    parser.add_argument('--midfolder', help='folder to store the intermediate files', type=str, default='temp/')
    parser.add_argument('--out', help='name of the output folder',  type=str, default = 'out/')
    parser.add_argument('--reject', help='threshold to reject contigs with a small fraction of proteins aligned.',  type=float, default = 0.3)
    parser.add_argument('--query_cover', help='The QC value set for DIAMOND BLASTP, setting to 0 means no query-cover constrain.',  type=int, default = 40)

    inputs = parser.parse_args()
    MAX_LENGTH = 500
    print('Running GiantHunter with the {} threads'.format(inputs.threads))
    #############################################################
    ######################  Check folders  ######################
    #############################################################

    out_fn = f'{inputs.out}/{inputs.midfolder}'

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
    
    run(inputs, out_fn, db_dir, MAX_LENGTH)
    print(f'GiantHunter finished, results are saved in {inputs.out}')

if __name__ == "__main__":
    main()