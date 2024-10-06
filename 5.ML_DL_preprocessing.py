##*************************************************************************##
##               Step8. extract variant calling from list file             ##          
##*************************************************************************##

#Must delete the plasmid genome

import pandas as pd
import sys

# from collections import defaultdict
# if len(sys.argv)!=4: 
#     print("Usage:python3 ML_DL_preprocessing.py <vcf snp counts> <feature to target file> <filtered feature to target file>  \
#     \n 
#     sys.exit()
# They will be used for the downstream deeplearning and ML training.


# classify the groups of treatments 
df_all_antibiotics = pd.read_csv('feature2target.txt',delimiter ='\t',header =0,).fillna(0)

#"(STP）S（≥17） I(13-16) R(≤12)" Spectinomycin
# "（FFC）S（≥18） I(13-17) R(≤12)"  Florfenicol
# "（CHL）S（≥18） I(13-17) R(≤12)"  chloramphenicol
# "(DOX) S（≥14） I(11-13) R(≤10)" Doxycycline 
# "（TGC)S（≥14） I(11-13) R(≤10)" Tigecycline
# "(CTX) S（≥26） I(23-25) R(≤22)" Cefotaxime
# "(CAZ) S（≥21） I(18-20) R(≤17)" Ceftazidime
# "CIP) S（≥26） I(22-25) R(≤21)" Ciprofloxacin
# "(OFX) S（≥16） I(13-15) R(≤12)" Ofloxacin
# "（POL) S（≥20） I(13-19) R(≤12)" Polymyxin B 
# "(ERY) S（≥23） I(14-22) R(≤13)" Erythromycin
# "(REP) S（≥20） I(17-19) R(≤16)" Rifampin
# "(TMP) S（≥16） I(11-15) R(≤10)" Trimethoprim
# (AMK） S（≥17） I(13-16) R(≤12)	 
#（TCY) S（≥14） I(11-13) R(≤10)

df_all_antibiotics.loc[df_all_antibiotics['Spectinomycin'] <= 12,'Spectinomycin']=-1
df_all_antibiotics.loc[(df_all_antibiotics['Spectinomycin'] >= 13) & (df_all_antibiotics['Spectinomycin'] <=16),'Spectinomycin']=0
df_all_antibiotics.loc[df_all_antibiotics['Spectinomycin'] >= 17,'Spectinomycin']=1

# df_all_antibiotics.loc[df_all_antibiotics['Florfenicol'] <= 12,'Florfenicol']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Florfenicol'] >= 13) & (df_all_antibiotics['Florfenicol'] <=17),'Florfenicol']=0
# df_all_antibiotics.loc[df_all_antibiotics['Florfenicol'] >= 18,'Florfenicol']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['chloramphenicol'] <= 12,'chloramphenicol']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['chloramphenicol'] >= 13) & (df_all_antibiotics['chloramphenicol'] <=17),'chloramphenicol']=0
# df_all_antibiotics.loc[df_all_antibiotics['chloramphenicol'] >= 18,'chloramphenicol']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['Doxycycline'] <= 10,'Doxycycline']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Doxycycline'] >= 11) & (df_all_antibiotics['Doxycycline'] <=13),'Doxycycline']=0
# df_all_antibiotics.loc[df_all_antibiotics['Doxycycline'] >= 14,'Doxycycline']=1

#df_all_antibiotics.loc[df_all_antibiotics['Tigecycline'] <= 10,'Tigecycline']=0
#df_all_antibiotics.loc[(df_all_antibiotics['Tigecycline'] >= 11) & (df_all_antibiotics['Tigecycline'] <=13),'Tigecycline']=1
#df_all_antibiotics.loc[df_all_antibiotics['Tigecycline'] >= 14,'Tigecycline']=2

df_all_antibiotics.loc[df_all_antibiotics['Cefotaxime'] <= 22,'Cefotaxime']=-1
df_all_antibiotics.loc[(df_all_antibiotics['Cefotaxime'] >= 23) & (df_all_antibiotics['Cefotaxime'] <=25),'Cefotaxime']=0
df_all_antibiotics.loc[df_all_antibiotics['Cefotaxime'] >= 26,'Cefotaxime']=1

df_all_antibiotics.loc[df_all_antibiotics['Ceftazidime'] <= 17,'Ceftazidime']=-1
df_all_antibiotics.loc[(df_all_antibiotics['Ceftazidime'] >= 18) & (df_all_antibiotics['Ceftazidime'] <=20),'Ceftazidime']=0
df_all_antibiotics.loc[df_all_antibiotics['Ceftazidime'] >= 21,'Ceftazidime']=1

# df_all_antibiotics.loc[df_all_antibiotics['Ciprofloxacin'] <= 21,'Ciprofloxacin']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Ciprofloxacin'] >= 22) & (df_all_antibiotics['Ciprofloxacin'] <=25),'Ciprofloxacin']=0
# df_all_antibiotics.loc[df_all_antibiotics['Ciprofloxacin'] >= 26,'Ciprofloxacin']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['Ofloxacin'] <= 12,'Ofloxacin']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Ofloxacin'] >= 13) & (df_all_antibiotics['Ofloxacin'] <=15),'Ofloxacin']=0
# df_all_antibiotics.loc[df_all_antibiotics['Ofloxacin'] >= 16,'Ofloxacin']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['Polymyxin_B'] <= 12,'Polymyxin_B']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Polymyxin_B'] >= 13) & (df_all_antibiotics['Polymyxin_B'] <=19),'Polymyxin_B']=0
# df_all_antibiotics.loc[df_all_antibiotics['Polymyxin_B'] >= 20,'Polymyxin_B']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['Erythromycin'] <= 13,'Erythromycin']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Erythromycin'] >= 14) & (df_all_antibiotics['Erythromycin'] <=22),'Erythromycin']=0
# df_all_antibiotics.loc[df_all_antibiotics['Erythromycin'] >= 23,'Erythromycin']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['Rifampin'] <= 16,'Rifampin']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Rifampin'] >= 17) & (df_all_antibiotics['Rifampin'] <=19),'Rifampin']=0
# df_all_antibiotics.loc[df_all_antibiotics['Rifampin'] >= 20,'Rifampin']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['Trimethoprim'] <= 10,'Trimethoprim']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Trimethoprim'] >= 11) & (df_all_antibiotics['Trimethoprim'] <=15),'Trimethoprim']=0
# df_all_antibiotics.loc[df_all_antibiotics['Trimethoprim'] >= 16,'Trimethoprim']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['amikacin'] <= 12,'amikacin']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['amikacin'] >= 13) & (df_all_antibiotics['amikacin'] <=16),'amikacin']=0
# df_all_antibiotics.loc[df_all_antibiotics['amikacin'] >= 17,'amikacin']=1
# 
# df_all_antibiotics.loc[df_all_antibiotics['Tetracycline'] <= 10,'Tetracycline']=-1
# df_all_antibiotics.loc[(df_all_antibiotics['Tetracycline'] >= 11) & (df_all_antibiotics['Tetracycline'] <=13),'Tetracycline']=0
# df_all_antibiotics.loc[df_all_antibiotics['Tetracycline'] >= 14,'Tetracycline']=1



df_all_antibiotics.set_index('locus_tag').to_csv('feature2target_processing.txt',sep = '\t')
