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

    #TODO doesn't support tiebreakers
    return pd.DataFrame(data={'name': resmat.index, 'win': wins, 'loss': losses, 'pct': pcts})\
        .sort_values('pct', ascending=False)\
        .drop('pct', axis=1)\
        .set_index('name')

# Return a results matrix where the given player wins every remaining game
def scenario_player_wins_out(resmat, player, ngames=6):
    unf = find_player_unfinished(resmat, player, ngames)
    spec_resmat = resmat.copy()
    spec_resmat.loc[player, unf.index] += unf
    return spec_resmat;

# Return a results matrix where the given player loses every remaining game
def scenario_player_loses_out(resmat, player, ngames=6):
    unf = find_player_unfinished(resmat, player, ngames)
    spec_resmat = resmat.copy()
    spec_resmat.loc[unf.index, player] += unf
    return spec_resmat;

# Return a list of all players who have guaranteed promotion
def promoting(resmat, ngames=6, nplayers=1):
    # If a player could lose all of their games and still promote, they've guaranteed promotion
    return [p for p in resmat.index if p in winloss(scenario_player_loses_out(resmat, p, ngames)).head(nplayers).index]

# Return a list of all players who have guaranteed demotion
def demoting(resmat, ngames=6, nplayers=1):
    # if a player could win all of their games and still demote, they've guaranteed demotion
    return [p for p in resmat.index if p in winloss(scenario_player_wins_out(resmat, p, ngames)).tail(nplayers).index]

# Generically, determine the players who, in some scenario, promote (or demote)
def possible(func, resmat, ngames=6, nplayers=1):
    players = func(resmat, ngames, nplayers)
    if len(players) >= nplayers:
        return players

    #FIXME until I find a better way, we use the O(n!) algorithm
    for player in resmat.index:
        for opponent, remaining_games in find_player_unfinished(resmat, player, ngames).items():
            for halfwins in range(int(2*(ngames - remaining_games)), 2*ngames+1):
                spec_resmat = resmat.copy()
                spec_resmat.loc[player, opponent] += halfwins / 2
                spec_resmat.loc[opponent, player] += remaining_games - halfwins / 2
                players.extend(func(spec_resmat, ngames, nplayers))

    return list(map(next, map(operator.itemgetter(1), itertools.groupby(sorted(players)))))

# Two specializations of possible(), one each for promotion and demotion
could_promote = functools.partial(possible, promoting)
could_demote  = functools.partial(possible, demoting)

with open(sys.argv[1]) as rfp:
    mat = matrix_from_results(rfp)

print(mat)
print(winloss(mat))
print(promoting(mat))
print(demoting(mat, nplayers=2))
print(could_promote(mat, nplayers=1))
print(could_demote(mat, nplayers=2))

