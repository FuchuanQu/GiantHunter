#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import pickle as pkl
import subprocess
import argparse
import shutil
import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
import  torch.utils.data as Data
from gianthunter.model import Transformer
from shutil import which
from collections import Counter
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from .scripts.ulity import *
from .scripts.preprocessing import *
from .scripts.parallel_prodigal_gv import main as parallel_prodigal_gv
from collections import defaultdict
import time
from tqdm import tqdm

# gianthunter --contigs example/test.fasta --out test/ --threads 8 --dbdir database



def run(inputs, db_dir, MAX_LENGTH=500):
    logger = get_logger()
    logger.info("Running program: GiantHunter (giant virus identification)")
    logger.info('Running GiantHunter with the {} threads'.format(inputs.threads))
    program_start = time.time()


    contigs   = inputs.contigs
    midfolder = inputs.midfolder
    out_dir   = 'final_prediction/'
    rootpth   = inputs.outpth
    db_dir    = inputs.dbdir
    threads   = inputs.threads


    if not os.path.isfile(contigs):
        exit()


    if not os.path.exists(db_dir):
        logger(f'Database directory {db_dir} missing or unreadable')
        exit(1)

    supplementary = 'supplementary'
    check_path(os.path.join(rootpth, out_dir))
    check_path(os.path.join(rootpth, out_dir, supplementary))
    check_path(os.path.join(rootpth, midfolder))



    ###############################################################
    #######################  Filter length ########################
    ###############################################################
    logger.info("[1/8] filtering the length of contigs...")
    rec = []
    genomes = {}
    for record in SeqIO.parse(contigs, 'fasta'):
        if len(record.seq) >= inputs.len:
            rec.append(record)
            genome = Genome()
            genome.id = record.id
            genome.length = len(record.seq)
            genome.genes = []
            genome.viral_hits = {}
            genome.regions = None
            genomes[genome.id] = genome
    # FLAGS: no contigs passed the length filter
    if not rec:
        with open(f'{rootpth}/{out_dir}/gianthunter_prediction.tsv', 'w') as file_out:
            file_out.write("Accession\tLength\tPotentialLineage\tScore\tGenus\tGenusCluster\n")
            for record in SeqIO.parse(contigs, 'fasta'):
                file_out.write(f'{record.id}\t{len({record.seq})}\tfiltered\t0\t-\t-\n')
        logger.info(f"GiantHunter finished! please check the results in {os.path.join(rootpth,out_dir, 'gianthunter_prediction.tsv')}")
        exit()


    _ = SeqIO.write(rec, f'{rootpth}/filtered_contigs.fa', 'fasta')

    ###############################################################
    ########################## Clustering  ########################
    ###############################################################
    if os.path.exists(f'{inputs.proteins}'):
        logger.info("[2/8] using provided protein file...")
        rec = []
        for record in SeqIO.parse(inputs.proteins, 'fasta'):
            genome_id = record.id.rsplit('_', 1)[0]
            try:
                _ = genomes[genome_id]
                rec.append(record)
            except:
                pass
        if rec:
            SeqIO.write(rec, f'{rootpth}/{midfolder}/query_protein.fa', 'fasta')
        else:
            logger.info("WARNING: no proteins found in the provided file.\nPlease check whether the genes is called by the prodigal.")
            logger.info("[2/8] calling genes with prodigal...")
            parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    elif os.path.exists(f'{rootpth}/{midfolder}/query_protein.fa'):
        logger.info("[2/8] reusing existing protein file...")
    else:
        logger.info("[2/8] calling genes with prodigal...")
        parallel_prodigal_gv(f'{rootpth}/filtered_contigs.fa', f'{rootpth}/{midfolder}/query_protein.fa', threads)
    
    # combine the database with the predicted proteins
    run_command(f"cat {db_dir}/RefVirus.faa {rootpth}/{midfolder}/query_protein.fa > {rootpth}/{midfolder}/ALLprotein.fa")
    # generate the diamond database
    run_command(f'diamond makedb --in {rootpth}/{midfolder}/query_protein.fa -d {rootpth}/{midfolder}/query_protein.dmnd --threads {threads} --quiet')
    # run diamond
    # align to the database
    if os.path.exists(f'{rootpth}/{midfolder}/db_results.tab'):
        logger.info("[3/8] using existing all-against-all alignment results...")
    else:  
        logger.info("[3/8] running all-against-all alignment...")
        run_command(f"diamond blastp --db {db_dir}/RefVirus.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/db_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
        run_command(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/db_results.tab > {rootpth}/{midfolder}/db_results.abc")
    # align to itself
    run_command(f"diamond blastp --db {rootpth}/{midfolder}/query_protein.dmnd --query {rootpth}/{midfolder}/query_protein.fa --out {rootpth}/{midfolder}/self_results.tab --outfmt 6 --threads {threads} --evalue 1e-5 --max-target-seqs 10000 --query-cover 50 --subject-cover 50 --quiet")
    run_command(f"awk '{{print $1,$2,$3,$12}}' {rootpth}/{midfolder}/self_results.tab > {rootpth}/{midfolder}/self_results.abc")

    logger.info("[4/8] generating networks...")
    # FLAGS: no proteins aligned to the database
    if os.path.getsize(f'{rootpth}/{midfolder}/db_results.abc') == 0:
        Accession = []
        Length_list = []
        for record in SeqIO.parse(f'{contigs}', 'fasta'):
            Accession.append(record.id)
            Length_list.append(len(record.seq))
        df = pd.DataFrame({"Accession": Accession, "Length": Length_list,  "PotentialLineage":['unknown']*len(Accession), "Score":[0]*len(Accession), "Genus": ['-']*len(Accession), "GenusCluster": ['-']*len(Accession)})
        df.to_csv(f"{rootpth}/{out_dir}/gianthunter_prediction.tsv", index = None, sep='\t')
        exit()

    # add the genome size
    genome_size = defaultdict(int)
    for index, r in enumerate(SeqIO.parse(f'{rootpth}/{midfolder}/ALLprotein.fa', 'fasta')):
        genome_id = r.id.rsplit('_', 1)[0]
        genome_size[genome_id] += 1



    ###############################################################
    ##########################  Prediction  #######################
    ###############################################################
    logger.info("[5/8] predicting the taxonomy...")
    contig2ORFs = {}
    for record in SeqIO.parse(f'{rootpth}/{midfolder}/query_protein.fa', "fasta"):
        contig = record.id.rsplit("_", 1)[0]
        try:
            contig2ORFs[contig].append(record.id)
        except:
            contig2ORFs[contig] = [record.id]

    contig_names = list([record.id for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta')])
    ORF2hits, all_hits = parse_alignment(f'{rootpth}/{midfolder}/db_results.abc')
    taxid2parent, taxid2rank = import_nodes(f'{db_dir}/nodes.csv')
    taxid2name = import_names(f'{db_dir}/names.csv')
    database_taxa_df = pd.read_csv(f'{db_dir}/taxid.csv', sep=",", header=None)
    database_taxa_df = database_taxa_df[database_taxa_df[0].isin(all_hits)]
    database2taxid = database_taxa_df.set_index(0)[1].to_dict()


    # List to store results for each contig
    results = []
    # Iterate over sorted contig names
    with tqdm(total=len(contig_names)) as pbar:
        for contig in sorted(contig_names):
            _ = pbar.update(1)
            # Check if contig has associated ORFs
            if contig not in contig2ORFs:
                # No ORFs found for this contig
                results.append([contig, "no ORFs found", ""])
                continue
            # Find LCA for each ORF in the contig
            LCAs_ORFs = [
                find_LCA_for_ORF(ORF2hits[ORF], database2taxid, taxid2parent)
                for ORF in contig2ORFs[contig]
                if ORF in ORF2hits
            ]
            # Check if any LCAs were found
            if not LCAs_ORFs:
                results.append([contig,"no hits to database", -1])
                continue
            # Find the weighted LCA for the contig
            lineages, lineages_scores = find_weighted_LCA(LCAs_ORFs, taxid2parent, 0.5)
            # Handle cases with no valid ORFs or lineages
            if lineages == "no ORFs with taxids found.":
                results.append([contig, "hits not found in taxonomy files", -1])
                continue
            if lineages == "no lineage larger than threshold.":
                results.append([contig, "no lineage larger than threshold.", -1])
                continue
            # Prepare lineage and score strings
            lineage_str = convert_lineage_to_names(lineages, taxid2name, taxid2rank)
            scores_str = ";".join(f"{score:.2f}" for score in lineages_scores)
            results.append([contig, lineage_str, scores_str])
            

    # Convert results to a DataFrame and save as CSV
    df = pd.DataFrame(results, columns=["Accession", "Lineage", "Score"])
    df['Length'] = df['Accession'].apply(lambda x: genomes[x].length if x in genomes else '-')


    Genus = [
        next((item.split(':')[1] for item in line.split(';') if item.startswith('genus:')), '-')
        if line not in {'no hits to database', 'no ORFs found', 'hits not found in taxonomy files', 'no lineage larger than threshold.'}
        else '-'
        for line in df['Lineage']
    ]

    df['Genus'] = Genus

    df.to_csv(f'{rootpth}/{midfolder}/taxonomy_prediction.tsv', index=False, sep='\t')

    ###############################################################
    ####################### dump results ##########################
    ###############################################################
    logger.info("[8/8] writing the results...")
    df = df.reset_index(drop=True)

    contigs_list = {item:1 for item in list(df['Accession'])}
    filtered_contig = []
    filtered_lenth = []
    unpredicted_contig = []
    unpredicted_length = []
    for record in SeqIO.parse(f'{contigs}', 'fasta'):
        try:
            _ = contigs_list[record.id]
        except:
            if len(record.seq) < inputs.len:
                filtered_contig.append(record.id)
                filtered_lenth.append(len(record.seq))
            else:
                unpredicted_contig.append(record.id)
                unpredicted_length.append(len(record.seq))


    # Create lists by combining existing data with new entries
    all_contigs = df['Accession'].tolist() + filtered_contig + unpredicted_contig
    all_pred = df['Lineage'].tolist() + ['filtered'] * len(filtered_contig) + ['-'] * len(unpredicted_contig)
    all_score = df['Score'].tolist() + [0] * len(filtered_contig) + [0] * len(unpredicted_contig)
    all_length = df['Length'].tolist() + filtered_lenth + unpredicted_length

    # Create DataFrame directly from lists
    contig_to_pred = pd.DataFrame({
        'Accession': all_contigs,
        'Length': all_length,
        'PotentialLineage': all_pred,
        'Score': all_score
    })

    ProkaryoticGroup = pkl.load(open(f'{db_dir}/ProkaryoticGroup.pkl', 'rb'))
    rows_to_update = contig_to_pred['PotentialLineage'].apply(lambda lineage: any(group in lineage for group in ProkaryoticGroup))
    # Update the pred_csv dataframe
    contig_to_pred.loc[rows_to_update, 'Prokaryotic virus (Bacteriophages and Archaeal virus)'] = 'Y'
    contig_to_pred.loc[~rows_to_update, 'Prokaryotic virus (Bacteriophages and Archaeal virus)'] = 'N'


    # Save DataFrame to CSV
    contig_to_pred.to_csv(f"{rootpth}/{midfolder}/taxonomy_prediction.tsv", index=False, sep='\t')

    # write the gene annotation results
    genes = load_gene_info(f'{rootpth}/{midfolder}/query_protein.fa', genomes)
    anno_df = pkl.load(open(f'{db_dir}/RefVirus_anno.pkl', 'rb'))
    
    # protein annotation
    df = pd.read_csv(f'{rootpth}/{midfolder}/db_results.tab', sep='\t', names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
    df = df.drop_duplicates('qseqid', keep='first').copy()
    df['coverage'] = df['length'] / df['qend']
    df.loc[df['coverage'] > 1, 'coverage'] = 1
    for idx, row in df.iterrows():
        gene = genes[row['qseqid']]
        try:
            gene.anno = anno_df[row['sseqid']]
        except:
            gene.anno = 'hypothetical protein'
        gene.pident = row['pident']
        gene.coverage = row['coverage']

    # write the gene annotation by genomes
    with open(f'{rootpth}/{out_dir}/{supplementary}/gene_annotation.tsv', 'w') as f:
        f.write('Genome\tORF\tStart\tEnd\tStrand\tGC\tAnnotation\tpident\tcoverage\n')
        for genome in genomes:
            for gene in genomes[genome].genes:
                f.write(f'{genome}\t{gene}\t{genes[gene].start}\t{genes[gene].end}\t{genes[gene].strand}\t{genes[gene].gc}\t{genes[gene].anno}\t{genes[gene].pident:.2f}\t{genes[gene].coverage:.2f}\n')


    #############################################################
    ####################  DIAMOND CREATEDB ######################
    #############################################################
    taxonomy_df = pd.read_csv(f'{rootpth}/{midfolder}/taxonomy_prediction.tsv', sep='\t')
    taxonomy_df['Score'] = taxonomy_df['Score'].apply(lambda x: x.split(';')[-1] if isinstance(x, str) else x)
    
    taxonomy_df['GiantVirus'] = taxonomy_df['PotentialLineage'].apply(lambda x: 'GiantVirus' if ('Nucleocytoviricota' in x or  \
                                                                               'Pandoravirus' in x or \
                                                                               'Pithoviruses' in x ) else 'Non-GiantVirus')
    GiantVirus_df = taxonomy_df[taxonomy_df['GiantVirus'] == 'GiantVirus']
    remained_df = taxonomy_df[(taxonomy_df['GiantVirus'] == 'Non-GiantVirus') & (taxonomy_df['Prokaryotic virus (Bacteriophages and Archaeal virus)'] == 'N')]
    if remained_df.empty:
        taxonomy_df = taxonomy_df[['Accession', 'Length', 'GiantVirus', 'PotentialLineage', 'Score' ]]
        non_gitant_df = taxonomy_df[taxonomy_df['GiantVirus'] == 'Non-GiantVirus']
        taxonomy_df.loc[non_gitant_df.index, 'PotentialLineage'] = '-'
        taxonomy_df.loc[non_gitant_df.index, 'Score'] = '-'
        taxonomy_df.loc[non_gitant_df.index, 'GiantVirus'] = 'Non-GiantVirus'

        taxonomy_df.to_csv(f'{rootpth}/{out_dir}/gianthunter_prediction.tsv', index=False, sep='\t')
        gene_annotation = pd.read_csv(f'{rootpth}/{out_dir}/{supplementary}/gene_annotation.tsv', sep='\t')
        gene_annotation = gene_annotation[gene_annotation['Genome'].isin(GiantVirus_df['Accession'].values)]
        gene_annotation.to_csv(f'{rootpth}/{out_dir}/{supplementary}/gene_annotation.tsv', index=False, sep='\t')
        rec = []
        check = {item: 1 for item in GiantVirus_df['Accession'].values}
        for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta'):
            try:
                _ = check[record.id]
                rec.append(record)
            except:
                pass
        SeqIO.write(rec, f'{rootpth}/{out_dir}/giant_virus_contigs.fa', 'fasta')

        rec = []
        for record in SeqIO.parse(f'{rootpth}/{midfolder}/query_protein.fa', 'fasta'):
            try:
                _ = check[record.id.rsplit('_', 1)[0]]
                rec.append(record)
            except:
                pass
        SeqIO.write(rec, f'{rootpth}/{out_dir}/{supplementary}/all_predicted_protein.fa', 'fasta')

        logger.info("GiantHunter finished! please check the results in {0}".format(os.path.join(rootpth, out_dir, 'gianthunter_prediction.tsv')))
        exit(0)

    rec = []
    check = {item: 1 for item in remained_df['Accession'].values}
    for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta'):
        try:
            _ = check[record.id]
            rec.append(record)
        except:
            pass
    
    SeqIO.write(rec, f'{rootpth}/{midfolder}/remained_contigs.fa', 'fasta')

    rec = []
    for record in SeqIO.parse(f'{rootpth}/{midfolder}/query_protein.fa', 'fasta'):
        try:
            _ = check[record.id.rsplit('_', 1)[0]]
            rec.append(record)
        except:
            pass
    SeqIO.write(rec, f'{rootpth}/{midfolder}/remained_protein.fa', 'fasta')


    diamond_db = f'{db_dir}/database.dmnd'

    try:
        if os.path.exists(diamond_db):
            logger.info(f'Using preformatted DIAMOND database ({diamond_db}) ...')
        else:
            # create database
            make_diamond_cmd = f'diamond makedb --threads {threads} --in {db_dir}/database.fa -d {db_dir}/database.dmnd'
            logger.info("Creating database...")
            _ = subprocess.check_call(make_diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        logger.info("diamond makedb failed")
        exit(1)

    try:
        # running alignment
        query_cover= inputs.query_cover
        if query_cover == 0:
            diamond_cmd = f'diamond blastp --threads {threads} --sensitive -d {diamond_db} -q {rootpth}/{midfolder}/remained_protein.fa -o {rootpth}/{midfolder}/results.tab -k 1 --evalue 1e-5 --quiet'
        else:
            diamond_cmd = f'diamond blastp --threads {threads} --sensitive -d {diamond_db} -q {rootpth}/{midfolder}/remained_protein.fa -o {rootpth}/{midfolder}/results.tab -k 1 --query-cover {query_cover} --evalue 1e-5 --quiet'
        logger.info("Running Diamond...")
        _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        diamond_out_fp = f"{rootpth}/{midfolder}/results.tab"
        database_abc_fp = f"{rootpth}/{midfolder}/results.abc"
        _ = subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
    except:
        logger.info("diamond blastp failed")
        logger.info(f"please try command {diamond_cmd}")
        exit(1)

    #############################################################
    ####################  Contig2Sentence  ######################
    #############################################################



    # Load dictonary and BLAST results
    proteins_df = pd.read_csv(f'{db_dir}/proteins.csv')
    proteins_df.dropna(axis=0, how='any', inplace=True)
    pc2wordsid = {pc: idx for idx, pc in enumerate(sorted(set(proteins_df['cluster'].values)))}
    protein2pc = {protein: pc for protein, pc in zip(proteins_df['protein_id'].values, proteins_df['cluster'].values)}
    blast_df = pd.read_csv(f"{rootpth}/{midfolder}/results.abc", sep=' ', names=['query', 'ref', 'evalue'])

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

    # propostion
    rec = []
    for key in blast_df['query'].values:
        name = key.rsplit('_', 1)[0]
        rec.append(name)
    counter = Counter(rec)
    mapped_num = np.array([counter[item] for item in id2contig.values()])

    rec = []
    for record in SeqIO.parse(f'{rootpth}/{midfolder}/remained_protein.fa', 'fasta'):
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
        logger.info("running the model on CPU")
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
        logger.info('cannot find pre-trained model')
        exit()

    ####################################################################################
    ##########################   Predit with contigs    ################################
    ####################################################################################

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
            pred  = ['GiantVirus' if item > 0.5 else 'Non-GiantVirus' for item in logit]
            all_pred += pred
            all_score += [float('{:.3f}'.format(i)) for i in logit]
            

    all_length = [genomes[id2contig[idx]].length for idx in id2contig]
    pred_csv = pd.DataFrame({"Accession":id2contig.values(), "Length": all_length, "GiantVirus":all_pred, "PotentialLineage": ["Unclassified"]*len(all_pred), "Score":all_score})

    # dump results
    taxonomy_df = taxonomy_df[['Accession', 'Length', 'GiantVirus', 'PotentialLineage', 'Score']]
    GiantVirus_df = GiantVirus_df[['Accession', 'Length', 'GiantVirus', 'PotentialLineage', 'Score']]
    pred_csv = pred_csv[pred_csv['GiantVirus'] == 'GiantVirus']
    merge_results = pd.concat((GiantVirus_df, pred_csv))
    non_gitant_df = taxonomy_df[~taxonomy_df['Accession'].isin(merge_results['Accession'].values)].copy()
    non_gitant_df['PotentialLineage'] = '-'
    non_gitant_df['Score'] = '-'
    non_gitant_df['GiantVirus'] = 'Non-GiantVirus'
    merge_results = pd.concat((merge_results, non_gitant_df))
    merge_results.to_csv(f'{rootpth}/{out_dir}/gianthunter_prediction.tsv', index=False, sep='\t')

    # write the supplementary files
    gene_annotation = pd.read_csv(f'{rootpth}/{out_dir}/{supplementary}/gene_annotation.tsv', sep='\t')
    gene_annotation = gene_annotation[gene_annotation['Genome'].isin(merge_results['Accession'].values)]
    gene_annotation.to_csv(f'{rootpth}/{out_dir}/{supplementary}/gene_annotation.tsv', index=False, sep='\t')
    rec = []
    check = {item: 1 for item in merge_results['Accession'].values}
    for record in SeqIO.parse(f'{rootpth}/filtered_contigs.fa', 'fasta'):
        try:
            _ = check[record.id]
            rec.append(record)
        except:
            pass
    SeqIO.write(rec, f'{rootpth}/{out_dir}/giant_virus_contigs.fa', 'fasta')

    rec = []
    for record in SeqIO.parse(f'{rootpth}/{midfolder}/query_protein.fa', 'fasta'):
        try:
            _ = check[record.id.rsplit('_', 1)[0]]
            rec.append(record)
        except:
            pass
    SeqIO.write(rec, f'{rootpth}/{out_dir}/{supplementary}/all_predicted_protein.fa', 'fasta')

    logger.info("GiantHunter finished! please check the results in {0}".format(os.path.join(rootpth, out_dir, 'gianthunter_prediction.tsv')))
    exit(0)
    

    logger.info("Run time: %s seconds\n" % round(time.time() - program_start, 2))


def main():
    parser = argparse.ArgumentParser(description="""GiantHunter is a python library for identifying NCLDVs from metagenomic data. 
                                    GiantHunter is based on a Transorfer model and relies on protein-based vocabulary to convert DNA sequences into sentences.""")
    parser.add_argument('--contigs', help='FASTA file of contigs',  default = 'test_contigs.fa')
    parser.add_argument('--proteins', help='FASTA file of predicted proteins (optional)', default = 'test/midfolder/query_protein.fa')
    parser.add_argument('--len', help='minimum length of contigs', type=int, default=3000)
    parser.add_argument('--threads', help='number of threads to use', type=int, default=int(os.cpu_count()))
    parser.add_argument('-d', '--dbdir', help='database directory (optional)',  default = 'database')
    parser.add_argument('--midfolder', help='folder to store the intermediate files', type=str, default='midfolder/')
    parser.add_argument('-o', '--outpth', help='name of the output folder',  type=str, default = 'out/')
    parser.add_argument('--reject', help='threshold to reject contigs with a small fraction of proteins aligned.',  type=float, default = 0.1)
    parser.add_argument('--query_cover', help='The QC value set for DIAMOND BLASTP, setting to 0 means no query-cover constrain.',  type=int, default = 40)

    inputs = parser.parse_args()
    MAX_LENGTH = 500
    
    
    run(inputs, MAX_LENGTH)

if __name__ == "__main__":
    main()