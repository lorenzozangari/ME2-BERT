
from sklearn.model_selection import StratifiedKFold
import pandas as pd





def train_test_val_split_stratified(df, train_frac=0.80, val_frac=0.1, domain_label='domain', seed=42):
    df_0 = df[df[domain_label] == .0].reset_index()
    df_1 = df[df[domain_label] == 1.0].reset_index()

    def split_group(group):
        group = group.sample(frac=1, random_state=seed)
        val_size = int(len(group) * val_frac)
        test_size = int(len(group) * (1 - train_frac - val_frac))

        val = group.iloc[:val_size]
        test = group.iloc[val_size:val_size + test_size]
        train = group.iloc[val_size + test_size:]
        return train, val, test

    train_0, val_0, test_0 = split_group(df_0)
    train_1, val_1, test_1 = split_group(df_1)

    test = pd.concat([test_0, test_1]).sample(frac=1, random_state=seed)
    return { 's_train': train_0, 't_train': train_1, 's_val':val_0, 't_val':val_1, 'test':test}



def k_fold_cross_validation_stratified(df, n_splits=5, domain_label='domain', seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    
    k_fold_splits = []
    
    for train_index, test_index in skf.split(df, df[domain_label]):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        # Further splitting train_df into train and validation
        train_0 = train_df[train_df[domain_label] == 0.0].reset_index(drop=True)
        train_1 = train_df[train_df[domain_label] == 1.0].reset_index(drop=True)
        
        val_frac = 1 / (n_splits - 1)
        
        def split_group(group):
            group = group.sample(frac=1, random_state=seed)
            val_size = int(len(group) * val_frac)
            val = group.iloc[:val_size]
            train = group.iloc[val_size:]
            return train, val
        
        train_split_0, val_split_0 = split_group(train_0)
        train_split_1, val_split_1 = split_group(train_1)
        
        s_train = train_split_0
        t_train = train_split_1
        s_val = val_split_0
        t_val = val_split_1
        test = test_df.sample(frac=1, random_state=seed)
        
        k_fold_splits.append({
            's_train': s_train,
            't_train': t_train,
            's_val': s_val,
            't_val': t_val,
            'test': test
        })
    
    return k_fold_splits