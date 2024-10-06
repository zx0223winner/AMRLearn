
from collections import defaultdict
import sys
import getopt

gene_file = ''
out_file = 'output.txt'
argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv, "hi:o:", ["input_file=", "output_file="])
except getopt.GetoptError as e:
	print(str(e) + '. Use KEGG.py -h to see argument options')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('KEGG.py -i <gene name file> -o <output file name>')
		print('or use KEGG.py --input_file=<gene name file> --output_file <output file name>')
		sys.exit()
	elif opt in ("-i", "--input_file"):
		gene_file = arg
	elif opt in ("-o", "--output_file"):
		out_file = arg
if gene_file == '':
	print("no gene name file.")
	sys.exit(2)
else:
	infile=open('ko00001-2.keg', 'r')
	lines = infile.readlines()
	c = ""
	o = ""
	k = ""
	ko_dic = {}
	gene_to_ko = defaultdict(list)
	for line in lines:
		line = line.strip('\n')
		if not line == "":
			if line.startswith('B'):
				if not line == 'B':
					line = line.replace('B  ', '')
					c = line
			elif line.startswith('C'):
				if not line == 'C':
					line = line.replace('C  ', '')
					o = line
			elif line.startswith('D'):
				if not line == 'D':
					line = line.replace('D      ', '')
					ks = line.split('  ')
					ko_name = ks[0]
					items = ' '.join(ks[1:]).split('; ')
					if len(items) > 1:
						genes, function = items[0], items[1]
					else:
						genes = items[0]
						function = ""
					k = function + '\t' + c + '\t' + o
					ko_dic[ko_name] = k
					for g in genes.split(', '):
						gene_to_ko[g].append(ko_name)
	infile.close()

	infile2 = open(gene_file, 'r')
	lines2 = infile2.readlines()
	outfile = open(out_file, 'w')
	for line2 in lines2:
		line2 = line2.strip('\n')
		if not line2 == "":
			output_line = line2
			gene_name = line2.split('\t')[0]
			if gene_name in gene_to_ko.keys():
				k, function, c, o = '', '', '', ''
				ko_list = list(set(gene_to_ko[gene_name]))
				for i in range(len(ko_list)):
					ko_name = ko_list[i]
					items = ko_dic[ko_name].split('\t')
					if i < len(ko_list)-1:
						k += ko_name + ', '
					else:
						k += ko_name
					if i == 0:
						function += items[0]
						c += items[1]
						o += items[2]
				output_line += '\t' + k + '\t' + function + '\t' + c + '\t' + o
			outfile.write(output_line + '\n')
	infile2.close()
	outfile.close()


