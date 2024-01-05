import logging
import pandas as pd
from typing import Sequence, List
import torch

def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def init_hidden_xavier(model: torch.nn.Module):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model


def split_train_val_test(df: pd.DataFrame, val_frac:float, test_frac:float = 0.0, stratify:Sequence = None)-> List[pd.DataFrame]:
    """Split a Dataframe to train, val and test based on fractions

    Args:
        df (pd.DataFrame): Dataframe with X and y
        val_frac (float): val fraction 
        test_frac (float): test fraction can be 0
        stratify (Sequence or None): Stratify sequence. Defaults to None.

    Returns:
        _type_: List of dataframe
    """
    from sklearn.model_selection import train_test_split
    ix = df.index
    eval_frac = test_frac + val_frac
    train_ix, eval_ix = train_test_split(ix, test_size=eval_frac, stratify=stratify)
    print(f"train_size =  {len(train_ix)} ")
    
    if test_frac>0:
        val_ix, test_ix = train_test_split(eval_ix, train_size=val_frac/eval_frac, stratify=stratify)
        print(f"val_size =  {len(val_ix)} ")
        print(f"test_size =  {len(test_ix)} ")
        return [df.loc[train_ix], df.loc[val_ix], df.loc[test_ix]]
    print(f"val_size =  {len(eval_ix)} ")
    return [df.loc[train_ix], df.loc[eval_ix]]

    
    

