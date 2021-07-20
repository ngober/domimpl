import sys
import re
import collections
import itertools
import functools
import operator
import numpy as np
import pandas as pd

def matrix_from_results(results, players=None):
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

# Find all unfinished matches for a given player.
def find_player_unfinished(resmat, player, ngames=6):
    remaining = ngames - ngames*np.eye(len(resmat.index)) - resmat - resmat.T
    return remaining.loc[player, remaining[player] > 0]

# Accumulate the win-loss statistics given a results matrix
def winloss(resmat):
    wins    = resmat.sum(axis=1)
    losses  = resmat.sum(axis=0)
    pcts    = wins / (wins + losses)

    if (wins.size > 2):
        htoh = pd.concat([winloss(resmat.loc[pcts == pct, pcts == pct])['win'] for pct in pcts.unique()])
    else:
        htoh = wins

    return pd.DataFrame(data={'name': resmat.index, 'win': wins, 'loss': losses, 'pct': pcts, 'neg_losses': -1*losses, 'htoh': htoh})\
        .sort_values(['pct', 'win', 'neg_losses', 'htoh'], ascending=False)\
        .drop(['pct', 'neg_losses', 'htoh'], axis=1)\
        .set_index('name')

def set_record(resmat, playerA, playerB, winsA, winsB):
    resmat.loc[playerA, playerB] = winsA
    resmat.loc[playerB, playerA] = winsB
    return resmat

def add_record(resmat, playerA, playerB, winsA, winsB):
    return set_record(resmat, playerA, playerB, resmat.loc[playerA, playerB] + winsA, resmat.loc[playerB, playerA] + winsB)

# Return a results matrix where the given player wins every remaining game
def scenario_player_wins_out(resmat, player, ngames=6):
    unf = find_player_unfinished(resmat, player, ngames)
    return add_record(resmat.copy(), player, unf.index, unf, 0)

# Return a results matrix where the given player loses every remaining game
def scenario_player_loses_out(resmat, player, ngames=6):
    unf = find_player_unfinished(resmat, player, ngames)
    return add_record(resmat.copy(), player, unf.index, 0, unf)

# Return a list of all players who have guaranteed promotion
def promoting(resmat, ngames=6, nplayers=1):
    # If a player could lose all of their games and still promote, they've guaranteed promotion
    return [p for p in resmat.index if p in winloss(scenario_player_loses_out(resmat, p, ngames)).head(nplayers).index]

# Return a list of all players who have guaranteed demotion
def demoting(resmat, ngames=6, nplayers=1):
    # if a player could win all of their games and still demote, they've guaranteed demotion
    return [p for p in resmat.index if p in winloss(scenario_player_wins_out(resmat, p, ngames)).tail(nplayers).index]

def could_promote(resmat, ngames=6, nplayers=1):
    players = promoting(resmat.copy(), ngames, nplayers)
    if len(players) >= nplayers:
        return players

    #FIXME until I find a better way, we use the O(n!) algorithm
    for player in resmat.index:
        for opponent, games in find_player_unfinished(resmat, player, ngames).items():
            players.extend(
                itertools.chain.from_iterable(
                    promoting(add_record(resmat.copy(), player, opponent, halfwins/2, games-halfwins/2)) for halfwins in range(int(2*(ngames - games)), 2*ngames+1)
                )
            )

    return list(map(next, map(operator.itemgetter(1), itertools.groupby(sorted(players)))))

def could_demote(resmat, ngames=6, nplayers=1):
    players = demoting(resmat, ngames, nplayers)
    if len(players) >= nplayers:
        return players

    #FIXME until I find a better way, we use the O(n!) algorithm
    for player in resmat.index:
        for opponent, games in find_player_unfinished(resmat, player, ngames).items():
            players.extend(
                itertools.chain.from_iterable(
                    demoting(add_record(resmat.copy(), player, opponent, halfwins/2, games-halfwins/2)) for halfwins in range(int(2*(ngames-games)+1))
                )
            )

    return list(map(next, map(operator.itemgetter(1), itertools.groupby(sorted(players)))))

with open(sys.argv[1]) as rfp:
    mat = matrix_from_results(rfp)

print(mat)
print(winloss(mat))
print(promoting(mat))
print(demoting(mat, nplayers=2))
print(could_promote(mat, nplayers=1))
print(could_demote(mat, nplayers=2))

