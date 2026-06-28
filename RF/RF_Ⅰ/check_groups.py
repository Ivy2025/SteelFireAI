import pandas as pd
from src.data_loader import load_data

X_train,X_test,y_train,y_test,features,groups_train,groups_test = load_data(
    'data/processed/steel_v2_cleaned_r2.csv',
    'fy_reduction',
    targets=['fy_reduction','fu_reduction','E_reduction']
)

# check group splitting
orig=pd.read_csv('data/processed/steel_v2_cleaned_r2.csv').replace(r'^\s+$', pd.NA, regex=True)
orig=orig.dropna(subset=['fy_reduction'])
orig=orig.drop(columns=['fu_reduction','E_reduction'],errors='ignore')
from sklearn.model_selection import GroupShuffleSplit
split=GroupShuffleSplit(test_size=0.2,random_state=42)
groups=orig['steel_ID']
for ti,tei in split.split(orig,orig['fy_reduction'],groups=groups):
    train_ids=groups.iloc[ti]
    test_ids=groups.iloc[tei]
print('train size',len(X_train),'test size',len(X_test))
print('intersection empty?', set(train_ids).intersection(set(test_ids)))
