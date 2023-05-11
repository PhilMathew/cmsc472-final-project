import pandas as pd
import random


def balanced_split(df, label_col, test_prop=0.2):
    pos_df, neg_df = df[df[label_col] == 1], df[df[label_col] == 0] # assuming binary classes
    pos_inds, neg_inds = list(pos_df.index), list(neg_df.index)
    
    # Restrict things so everything is balanced on the limiting class
    if min(len(pos_inds), len(neg_inds)) == len(pos_inds):
        random.shuffle(neg_inds)
        neg_inds = neg_inds[:len(pos_inds)]
    else:
        random.shuffle(pos_inds)
        pos_inds = pos_inds[:len(neg_inds)]
    
    # Find limiting class and sample based on that
    num_samples = int(test_prop * min(len(pos_inds), len(neg_inds)))
    testval_pos_inds = random.sample(pos_inds, k=num_samples)
    testval_neg_inds =  random.sample(neg_inds, k=num_samples)
    
    # Split val and test apart
    test_pos_inds, val_pos_inds = testval_pos_inds[:(len(testval_pos_inds) // 2)], testval_pos_inds[((len(testval_pos_inds) // 2) + 1):]
    test_neg_inds, val_neg_inds = testval_neg_inds[:(len(testval_pos_inds) // 2)], testval_neg_inds[((len(testval_neg_inds) // 2) + 1):]
    test_inds = test_pos_inds + test_neg_inds
    val_inds = val_pos_inds + val_neg_inds
    
    # Split into test and train
    test_df = df[df.index.isin(test_inds)]
    val_df = df[df.index.isin(val_inds)]
    train_df = df[(~df.index.isin(test_inds + val_inds)) & df.index.isin(neg_inds + pos_inds)]   
    
    return train_df, val_df, test_df
    


def main():
    data_df = pd.read_csv('data_csvs/data.csv')
    train_df, val_df, test_df = balanced_split(data_df, label_col='AFIB', test_prop=0.2)
    train_df.to_csv('data_csvs/train.csv')
    val_df.to_csv('data_csvs/val.csv')
    test_df.to_csv('data_csvs/test.csv')


if __name__ == '__main__':
    main()
    