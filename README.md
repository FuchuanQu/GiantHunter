
![GiantHunter](logo.jpeg)


# Overview
The main function of GiantHunter is to identify NCLDV-like contigs from metagenomic data. The input of the program should be fasta files and the output will be a csv file showing the predictions. 

If you have any trouble installing or using GiantHunter, you can post an issue or directly email us (jiayushang@cuhk.edu.hk). We welcome any suggestions.


### Quick install
*Note*: we suggest you to install all the package using conda (both miniconda and [Anaconda](https://anaconda.org/) are ok).

### Prepare the environment
1. When you use GiantHunter at the first time
```
# if you are a user of PhaBOX2 (https://github.com/KennthShang/PhaBOX)
# You can skip this installation of the phabox2 environment.
conda create -n phabox2 phabox=2.1.11 -c conda-forge -c bioconda -y
conda activate phabox2

# Download the scripts of GiantHunter
git clone https://github.com/FuchuanQu/GiantHunter.git
cd GiantHunter/
pip install .
cd ..

# Download the database
wget https://github.com/FuchuanQu/GiantHunter/releases/download/v2.0/gianthunter_db_v1.zip
unzip gianthunter_db_v1.zip

# run an example
gianthunter --contigs GiantHunter/example/test.fasta --d gianthunter_db_v1/ --o test --threads 4
# You can remove the GiantHunter if you want, but please keep the database :)
rm -rf GiantHunter/
```


2. If the example can be run without any bugs, you only need to activate your environment before using GiantHunter.
```
conda activate phabox2
```

If you cannot download the database in mainland China, please try the link below:
```
https://pan.baidu.com/s/1YJBaXA0OmvUfXEK8QEnkDg
pwd:jrif 
```

## Usage

```
gianthunter [--contigs INPUT_FA] [--out OUTPUT_DR] [--reject THRESHOLD] [--midfolder DIR] [--threads NUM] [--dbdir DR] [--query_cover QC]
```

**Options**


      --contigs INPUT_FA
                            input fasta file
      --len MINIMUM_LEN
                            predict only for sequence >= len bp (default 3000)
      --proteins PROTEIN_FA
                            An optional protein file. If you have already annotated your contigs, you can use them as the inputs. 
                            Otherwise, GiantHunter will run prodigal to translate your contigs.
      --threads NUM
                            Number of threads to run GiantHunter (default maximum can be allocated)
      --dbdir/-d DATABASE_DR
                            An optional path to store the database directory (default database/)
      --out/-o OUTPUT_DR
                            The output csv file (prediction)
      --reject THRESHOLD
                            Threshold to reject contigs with a small fraction of proteins aligned. (default 0.3)
      --midfolder TEMP_DIR
                            Folder to store the intermediate files (default temp/)
      --query_cover QC
                            The QC value set for DIAMOND BLASTP, setting to 0 means no query-cover constrain (default 40) 

**Example**

Prediction on the example file:

    gianthunter --contigs GiantHunter/example/test.fasta --o test/  --d gianthunter_db_v1 --threads 8

Please always remember to input the correct path where you store the GiantHunter database.

### Outputs

The main outputs in the folder [--out OUTPUT_DR]:

```
final_prediction
├── gianthunter_prediction.tsv
├── giant_virus_contigs.fa
└── supplementary
    ├── all_predicted_protein.fa    || Giant virus proteins predicted by prodigal-gv
    └── gene_annotation.tsv         || Gene annotation of the giant virus

```
    
    
### References

The paper is accepted by ISMB/ECCB 2025. See you in Liverpool!

The arXiv version can be found via: [arXiv version](https://arxiv.org/abs/2501.15472)

## You may also want to try our [PhaBOX webserver](https://phage.ee.cityu.edu.hk/)!

PhaBOX has now been upgraded to version 2.0 with faster speed!

There are some major components, including:

🎉 Generalized for all kinds of viruses; more than just bacteriophage

🎉 Virus identification (latest PhaMer)

🎉 Taxonomy classification (latest PhaGCN)

🎉 Host prediction (latest CHERRY/HostG)

🎉 Lifestyle prediction (latest PhaTYP)

🎉 Contamination/provirus detection

🎉 vOTU grouping

🎉 Phylogenetic tree based on marker genes

🎉 Viral protein annotation

🎉 All the databases are updated to the latest ICTV 2024 release

The local version can also be accessed via: https://github.com/KennthShang/PhaBOX. Feel free to check it out! 


