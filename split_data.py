import random
from argparse import ArgumentParser

from pathlib import Path
import pandas as pd


def shuffle_df(df):
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


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
    
    # Shuffle the dataframes
    train_df = shuffle_df(train_df)
    val_df = shuffle_df(val_df)
    test_df = shuffle_df(test_df)
    
    return train_df, val_df, test_df
    


def main():
    parser = ArgumentParser(description="Balanced Data Splitting Script")
    parser.add_argument('--input_csv', dest='input_csv', help='Path to CSV containing dataset paths and labels')
    parser.add_argument('--dx_type', dest='dx_type', default='SB', help='Diagnosis code for disease of interest in ConditionNames_SNOMED-CT.csv')
    parser.add_argument('--test_prop', dest='test_prop', default=0.2, help='Proportion of data to split out as a test set')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='data_csvs', help='Path to directory to output CSVs to')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    
    data_df = pd.read_csv(args.input_csv)
    train_df, val_df, test_df = balanced_split(data_df, label_col=args.dx_type, test_prop=0.2)
    train_df.to_csv(str(output_dir / 'train.csv'))
    val_df.to_csv(str(output_dir / 'val.csv'))
    test_df.to_csv(str(output_dir / 'test.csv'))


if __name__ == '__main__':
    main()
    