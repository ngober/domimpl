import sys
import re
import collections
import pandas as pd
import itertools
import operator
import numpy as np

def matrix_from_results(results):
    matrix = collections.defaultdict(dict)
    for res in results:
        match = re.match(r'([\w ]+) (\d(?:\.\d)?)–(\d(?:\.\d)?) ([\w ]+)', res)
        if match is not None:
            matrix[match.group(4)][match.group(1)] = float(match.group(2))
            matrix[match.group(1)][match.group(4)] = float(match.group(3))
            matrix[match.group(1)][match.group(1)] = 0.
            matrix[match.group(4)][match.group(4)] = 0.

    return pd.DataFrame.from_records(matrix).fillna(0.).sort_index(axis=0).sort_index(axis=1)

def find_unfinished(resmat, ngames=6):
    idx = (np.triu(np.ones(resmat.shape)) - np.eye(len(resmat.index))) * ngames
    played = resmat + resmat.T
    remaining = (idx - played).where(played < idx)
    remaining = remaining.stack().reset_index()
    remaining.columns = ['player1', 'player2', 'Games']
    return remaining

def winloss(resmat):
    records = [(resmat.loc[name].sum(), resmat.loc[:,name].sum(), name) for name in resmat.index]
    return sorted(records, key=lambda x: x[1]/(x[0]+x[1])) # sort by ascending loss percent

def could_demote(resmat, ndemote=1):
    pass

def could_promote(resmat, npromote=1):
    pass

with open(sys.argv[1]) as rfp:
    mat = matrix_from_results(rfp)
    print(mat)
    print(find_unfinished(mat))
    #for record in winloss(mat):
        #print(record)

