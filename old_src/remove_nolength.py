from Bio import SeqIO
for current_seq in SeqIO.parse("clustering/fasta_merged/fasta_merged.fasta", "fasta"):
    if current_seq.seq != '' and len(current_seq.seq)<1000:
        print (">" + current_seq.id + "\n" + current_seq.seq)
