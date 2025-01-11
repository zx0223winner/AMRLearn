##*************************************************************************##
##   Step2. parse the variant calling file and create SNPs counts matrix   ##          
##*************************************************************************##

import sys
from collections import defaultdict
if len(sys.argv)!=4: 
    print("Usage:python3 vcf2snp.py <gene location info> <parsnp vcf file> <output file>  \n \n e.g., python3 gene_location_info.txt parsnp.ggr.vcf vcf_snp_count.txt")
    sys.exit()

handle = open(sys.argv[1],'r')
lines = handle.read().split('\n')[1:]
handle1 = open(sys.argv[2],'r')
line1s = handle1.read().split('\n')
a = ''
b = ''
c = ''
d = ''
e = ''
# my_dict={}
my_dict1={}
name_list = []
name_dict = defaultdict(list)
cur_line = 0
while cur_line < len(line1s):
    if line1s[cur_line].startswith('#CHROM'):
        items = line1s[cur_line].split('\t')
        for i in range(9, len(items)):
            name = items[i].split('.')[0]
            name_list.append(name)
            name_dict[name] = []
        break
    cur_line += 1
np_lines = [line for line in lines if line.strip() != ""]
for line in np_lines:
#     if line.startswith('EC'):
    a,b,c,d,e = line.split('\t')
#         my_dict[a] = b +'\t'+c+'\t'+d + '\t' + e
    #while cur_line < len(line1s):
    for line1 in line1s:
        if line1.startswith('U'):
            list = line1.split('\t')
            if int(list[1]) > int(d):
                break
            if list[7][:-4] != 'SYN' and int(list[1]) >= int(c):
                   # D96_6_1_5 = D96_6_1_5 + list[10]
                for i in range(len(name_list)):
                    name_dict[name_list[i]].append(int(list[(9+i)]))
        cur_line += 1
    outline = b+'\t'+c+'\t'+d+'\t'+e
    for i in range(len(name_list)):
        outline += '\t' + str(len([value for value in name_dict[name_list[i]] if int(value) > 0])/int(e))
        #sum(name_dict[name_list[i]]
        #print(len([value for value in name_dict[name_list[i]] if int(value) > 0]))
    #print(name_dict[name_list[1]])
    
    my_dict1[a] = outline
    for i in range(len(name_list)):
            name_dict[name_list[i]] = []
        #print(a + '\t'+ my_dict[a] + '\t' + my_dict1[a])

outfile=open(sys.argv[3],'w')
title = 'locus_tag'+'\t'+'gene_name'+'\t'+'start_site'+'\t'+'end_site'+'\t'+'gene_length'
for i in range(len(name_list)):
    title += '\t' + name_list[i]
outfile.write(title + '\n')
for linen in my_dict1.keys():
    outfile.write(linen + '\t'+ my_dict1[linen]+'\n')
    #outfile.write(linen + '\t'+ my_dict[a] + '\t'+ my_dict1[linen]+'\n')
     
handle.close()
handle1.close()



