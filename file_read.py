import pandas as pd
import collections
import re
import functools

def from_DomBot_matches(results, players=None):
    if players is not None:
        matrix = {p : { q : 0 for q in players } for p in players }
    else:
        matrix = collections.defaultdict(functools.partial(collections.defaultdict, int))

    for res in results:
        match = re.match(r'([\w ]+) (\d(?:\.\d)?)–(\d(?:\.\d)?) ([\w ]+)', res)
        if match is not None:
            matrix[match.group(4)][match.group(1)] += float(match.group(2))
            matrix[match.group(1)][match.group(4)] += float(match.group(3))

    return pd.DataFrame.from_records(matrix).fillna(0.).sort_index(axis=0).sort_index(axis=1)

