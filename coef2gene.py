
import sys
if len(sys.argv)!=4: #if the input arguments not 4, showing the usage.
	print("Usage:python3 index.py <the features file> <list file> <outputfile> \n ")
	sys.exit()

file1=open(sys.argv[1],'r').read().split('\n')
outfile = open(sys.argv[3],'w')
file2=open(sys.argv[2],'r').read().split('\n')
dict = {}
for line1 in file1:
	if line1 !='':
		handle1=line1.split('\t')
		dict[handle1[0]]=handle1[1]
outfile.write('locus_tag\tgene_name\tcoef\n')
for line2 in file2:
	if line2 !='':
		handle2 =line2.split('\t')
		if handle2[0] in dict.keys():
			outfile.write(handle2[0]+'\t'+dict.get(handle2[0])+'\t'+handle2[1]+'\n')