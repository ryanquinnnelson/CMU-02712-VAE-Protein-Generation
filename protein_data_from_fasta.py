import csv
from Bio import SeqIO

# Sequences taken from UniProtKB database searching for PABP protein names with specified length
# number of sequences with length 100-300: 264
# number of sequences with length 100-1000: 6010


def txt_files_from_fasta(fasta_file):
    '''
    Takes a .fasta file as input and creates a .txt file for each entry that contains
    the protein sequence.

    :param fasta_file: a .fasta file
    :return: none
    '''

    with open(fasta_file) as file:
        for record in SeqIO.parse(file, "fasta"):
            file_name_list = str(record.id).split("|")
            file_name = "_".join(file_name_list)
            with open(f"protein_sequences/{file_name}.txt", "w") as protein_file:
                # each .txt file written to the protein_sequences folder
                protein_file.write(str(record.seq))


def csv_from_fasta(fasta_file, output_file):
    '''
    Takes a .fasta file as input and creates a single .csv file containing all
    of the protein sequences contained in the .fasta file. Each row of the .csv
    file corresponds to one sequence in the .fasta file.

    :param fasta_file: a .fasta file; the input .fasta file
    :param output_file: a .csv file; the name of the .csv file to be created
    :return: none
    '''

    with open(output_file, "w", newline="") as output:
        csvwriter = csv.writer(output)
        with open(fasta_file) as file:
            for record in SeqIO.parse(file, "fasta"):
                sequence = [str(record.seq)]
                csvwriter.writerow(sequence)


# If individual .txt files desired, use these functions
# txt_files_from_fasta("pabp_100-300.fasta")
# txt_files_from_fasta("pabp_100-1000.fasta")

csv_from_fasta("pabp_100-300.fasta", "protein_sequences_100-300.csv")
csv_from_fasta("pabp_100-1000.fasta", "protein_sequences_100-1000.csv")