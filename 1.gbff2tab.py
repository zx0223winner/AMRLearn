##*************************************************************************##
##          Step1. extract gene location info from GenBank file            ##          
##*************************************************************************##


import sys
if len(sys.argv)!=3: 
    print("Usage:python3 gbff2tab.py <gene_bank_file> <gene_output_file>  \n \n e.g., python3 GCF_003790525.1_genomic.gbff gene_location_info.txt")
    sys.exit()

handle = open(sys.argv[1],'r')
lines = handle.read().split('\n')
my_dict={}
a = ''
b = ''
c = ''
d = ''
for line in lines:
    if line.startswith("                     /locus_tag") and a != '':
        d = line.split('"')[1].strip('"')
    if line.startswith("     gene"):
        list = line.split('            ')[1]
        (a,b) = list.split('..')[:2]
        if a.startswith('complement(join'):
            a = a.split('join(')[1]
            b = b.split(',')[0]
        elif a.startswith('complement'):
            a = a.split('(')[1]
            b = b.split(')')[0]
        elif a.startswith('join'):
            if ',' in b:
                a = a.split('(')[1]
                b = b.split(',')[0]
            else: 
                a = a.split('(')[1]
                b = b.split(')')[0]
        a = a.strip('>').strip('<')
        b = b.strip('>').strip('<')

    if line.startswith("                     /gene=") and a != '':
        c = line.split('"')[1].strip('"')
    if d != '':
        my_dict[d]=c + '\t'+ a +'\t'+ b + '\t' + str((int(b)-int(a)+1)) # add  the gene length; TypeError: unsupported operand type(s) for -: 'str' and 'str'; using int and str
        a = ''
        b = ''
        c = ''
        d = ''
handle.close()

outfile=open(sys.argv[2],'w')
outfile.write('locus_tag'+'\t'+'gene_name'+'\t'+'start_site'+'\t'+'end_site'+'\t'+'gene_length'+'\n')
for line1 in my_dict.keys():
     outfile.write(line1+"\t"+my_dict[line1]+'\n')


