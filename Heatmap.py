import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import sys

row_size = 20
col_size = 20

# prepare data
heatmap_data = pd.read_csv(sys.argv[1], sep='\t', header=None,
                           names=['gene name', 'Cefotaxime', 'Ceftazidime', 'Spectinomycin',
                                   'ko_num', 'description', 'category', 'function'])
heatmap_data = heatmap_data.tail(-1)
heatmap_data = heatmap_data.drop(columns=['description']) #'ALL', 
heatmap_data = heatmap_data[heatmap_data['ko_num'].notna()]
heatmap_data = heatmap_data.set_index(['gene name', 'ko_num', 'function', 'category'])
heatmap_data.reset_index(level=['category'], inplace=True)
# threshold = 2
# if len(selected) > 3:
#     threshold = 3
# heatmap_data = heatmap_data.dropna(thresh=threshold)
heatmap_data = heatmap_data.fillna(0)
heatmap_data['Cefotaxime'] = heatmap_data['Cefotaxime'].astype(str).astype(int)
heatmap_data['Ceftazidime'] = heatmap_data['Ceftazidime'].astype(str).astype(int)
heatmap_data['Spectinomycin'] = heatmap_data['Spectinomycin'].astype(str).astype(int)

heatmap_data.sort_values(by='category', inplace=True)
category = heatmap_data.pop("category")
lut = dict(zip(category.unique(), cls.CSS4_COLORS))
print(lut)
row_colors = category.map(lut)
cmap = sns.cubehelix_palette(dark=.25, light=.75, as_cmap=True)
cm = sns.clustermap(heatmap_data, robust=True, figsize=(row_size, col_size), dendrogram_ratio=[0.2, 0.25],
                    row_cluster=False, col_cluster=False, row_colors=row_colors, cmap=cmap, center=1)
plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
save_name = 'heatmap_' + sys.argv[1] + '.eps'
plt.savefig(save_name)
