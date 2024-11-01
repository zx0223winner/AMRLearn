#counting the percentage of non-coding regions

# For example,
#non_coding_1	region	1	189	189
#non_coding_2	region	256	337	81
#non_coding_3	region	2800	2801	1
#non_coding_4	region	3733	3734	1

import sys

if len(sys.argv)!=2: 
    print("Usage:python3 non-coding_gene_location.py <gene_location_info.txt> ")
    sys.exit()

handle = open(sys.argv[1],'r')
lines = handle.read().split('\n')[1:]
start =[]
end = []
np_lines = [line for line in lines if line.strip() != ""] # ValueError: not enough values to unpack (expected 5, got 1)
for line in np_lines:
    (a,b,c,d,e)=line.split('\t')
    if c < d:
        start.append(c)
        end.append(d)
    else:
        start.append(d)
        end.append(c)
i=0
length =0
outfile = open('non-coding_gene_location_info.txt','w')
outfile.write('locus_tag'+'\t'+'gene_name'+'\t'+'start_site'+'\t'+'end_site'+'\t'+'gene_length'+'\n')
for i in range(len(start)-1):
    if (int(start[i+1]) - int(end[i])) >= 0:
        length = int(start[i+1]) - int(end[i])
        outfile.write('non_coding_'+ str(i+1) +'\t'+ 'region'+'\t'+str(int(end[i])+1) +'\t'+ str(int(start[i+1]))+'\t' + str(length)+'\n')

outfile.close()
handle.close()
