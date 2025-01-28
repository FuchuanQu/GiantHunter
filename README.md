
![GiantHunter](logo.jpeg)


# Overview
The main function of GiantHunter is to identify NCLDV-like contigs from metagenomic data. The input of the program should be fasta files and the output will be a csv file showing the predictions. Since it is a deep learning model, if you have GPU units on your PC, we recommend that you use them to save time. 

If you have any trouble installing or using GiantHunter, please let us know by emailing us (qu.fuchuan@my.cityu.edu.hk).


## Required Dependencies
* Python 3.x
* Numpy
* Pandas
* Pytorch>1.8.0
* [Diamond](https://github.com/bbuchfink/diamond)
* [Prodigal](https://github.com/hyattpd/Prodigal)


If you want to use the gpu to accelerate the program:
* cuda
* Pytorch-gpu

* For cpu version pytorch: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
* For gpu version pytorch: Search [pytorch](https://pytorch.org/) to find the correct cuda version according to your computer

### Quick install
*Note*: we suggest you to install all the package using conda (both miniconda and [Anaconda](https://anaconda.org/) are ok).

After cloning this respository, you can use anaconda to install the **GiantHunter.yaml**. This will install all packages you need with gpu mode (make sure you have installed cuda on your system to use the gpu version. Othervise, it will run with cpu version). The command is: `conda env create -f GiantHunter.yaml -n gianthunter`


### Prepare the environment
1. When you use GiantHunter at the first time
```
cd GiantHunter/
conda env create -f GiantHunter.yaml -n gianthunter
conda activate gianthunter
```


2. If the example can be run without any bugs, you only need to activate your 'gianthunter' environment before using GiantHunter.
```
conda activate gianthunter
```


## Usage

```
python run.py [--contigs INPUT_FA] [--out OUTPUT_CSV] [--reject THRESHOLD] [--midfolder DIR] [--threads NUM] [--dbdir DR] [--query_cover QC]
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
                            Number of threads to run GiantHunter (default 8)
      --dbdir DR
                            An optional path to store the database directory (default database/)
      --out OUTPUT_CSV
                            The output csv file (prediction)
      --reject THRESHOLD
                            Threshold to reject prophage. The higher the value, the more prophage will be rejected (default 0.3)
      --midfolder DIR
                            Folder to store the intermediate files (default gianthunter/)
      --query_cover QC
                            The QC value set for DIAMOND BLASTP, setting to 0 means no query-cover constrain (default 40) 

**Example**

Prediction on the example file:

    python run.py --contigs test/test.fasta --midfolder test/temp --out test/prediction.csv

The prediction will be written in *prediction.csv*, while the filtered NCLDV contigs will be written in *{midfolder}/output_ncldv_contigs.fa*. The CSV file has three columns: contigs names, prediction, and prediction score. The test_contig.fasta contains some NCLDV contigs, so the output is almost all NCLDV.
    
### References

The arXiv version can be found via: [arXiv version]()

### Contact
If you have any questions, please email us: qu.fuchuan@my.cityu.edu.hk
