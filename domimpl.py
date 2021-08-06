import itertools
import functools
import operator
import numpy as np
import pandas as pd

from img_draw import impl_image
import file_read

def find_unfinished(resmat, ngames=6):
    remaining = ngames*np.triu(np.ones(resmat.shape), 1) - resmat - resmat.T
    return remaining[remaining > 0].stack()

# Find all unfinished matches for a given player.
def find_player_unfinished(resmat, player, ngames=6):
    unf = find_unfinished(resmat, ngames)
    unfA = unf.loc[player] if player in unf.index else pd.Series(dtype=int)
    unf = unf.swaplevel()
    unfB = unf.loc[player] if player in unf.index else pd.Series(dtype=int)
    return pd.concat((unfA, unfB))

# Accumulate the win-loss statistics given a results matrix
def winloss(resmat):
    wins    = resmat.sum(axis=1)
    losses  = resmat.sum(axis=0)
    pcts    = wins / (wins + losses)
    htoh    = pd.concat([winloss(resmat.loc[pcts == pct, pcts == pct])['win'] for pct in pcts.unique() if pct > 0]) if pcts.unique().size > 1 else wins

    return pd.DataFrame(data={'name': resmat.index, 'win': wins, 'loss': losses, 'pct': pcts, 'neg_losses': -1*losses, 'htoh': htoh})\
        .sort_values(['pct', 'win', 'neg_losses', 'htoh'], ascending=False)\
        .drop(['pct', 'neg_losses', 'htoh'], axis=1)\
        .set_index('name')

def set_record(resmat, playerA, playerB, winsA, winsB):
    newval = pd.DataFrame(0, index=resmat.index, columns=resmat.columns)
    newval.loc[playerA, playerB] = winsA
    newval.loc[playerB, playerA] = winsB
    retmask = pd.DataFrame(False, index=resmat.index, columns=resmat.columns, dtype=bool)
    retmask.loc[playerA, playerB] = True
    retmask.loc[playerB, playerA] = True
    return resmat[~retmask] + newval[retmask]

def add_record(resmat, playerA, playerB, winsA, winsB):
    retval = pd.DataFrame(0, index=resmat.index, columns=resmat.columns)
    retval.loc[playerA, playerB] = winsA
    retval.loc[playerB, playerA] = winsB
    return resmat + retval

# Return a results matrix where the given player wins every remaining game
def scenario_player_wins_out(resmat, player, ngames=6):
    unf = find_player_unfinished(resmat, player, ngames)
    return add_record(resmat, player, unf.index, unf, 0)

# Return a results matrix where the given player loses every remaining game
def scenario_player_loses_out(resmat, player, ngames=6):
    unf = find_player_unfinished(resmat, player, ngames)
    return add_record(resmat, player, unf.index, 0, unf)

# Return a list of all players who have guaranteed promotion
def promoting(resmat, ngames=6, nplayers=1):
    # If a player could lose all of their games and still promote, they've guaranteed promotion
    return [ p for p in resmat.index if p in winloss(scenario_player_loses_out(resmat, p, ngames)).head(nplayers).index ]

# Return a list of all players who have guaranteed demotion
def demoting(resmat, ngames=6, nplayers=1):
    # if a player could win all of their games and still demote, they've guaranteed demotion
    return [ p for p in resmat.index if p in winloss(scenario_player_wins_out(resmat, p, ngames)).tail(nplayers).index ]

def could_promote(resmat, ngames=6, nplayers=1):
    # Resolve one match at a time, iteratively
    players = []

    for p in resmat.index:
        spec_resmat = scenario_player_wins_out(resmat, p, ngames)
        promo = promoting(spec_resmat, ngames, nplayers)
        players.extend(promo)

        if len(promo) < nplayers:
            for q,r in find_unfinished(spec_resmat, ngames).items():
                promo = [could_promote(add_record(spec_resmat, *q, hw/2, r-hw/2), ngames, nplayers) for hw in range(int(2*r+1))]
                players.extend(itertools.chain.from_iterable(promo))

    return list(map(next, map(operator.itemgetter(1), itertools.groupby(sorted(players)))))

def could_demote(resmat, ngames=6, nplayers=1):
    # Resolve one match at a time, iteratively
    players = []

    for p in resmat.index:
        spec_resmat = scenario_player_loses_out(resmat, p, ngames)
        demo = demoting(spec_resmat, ngames, nplayers)
        players.extend(demo)

        if len(demo) < nplayers:
            for q,r in find_unfinished(spec_resmat, ngames).items():
                demo = [could_demote(add_record(spec_resmat, *q, hw/2, r-hw/2), ngames, nplayers) for hw in range(int(2*r+1))]
                players.extend(itertools.chain.from_iterable(demo))

    return list(map(next, map(operator.itemgetter(1), itertools.groupby(sorted(players)))))

def match_impl(resmat, playerA, playerB, ngames=6, nplayers=1):
    unf = find_player_unfinished(resmat, playerA)
    remaining = unf[playerB] if playerB in unf.index else 0

    # Gather data about the implications of scenarios
    resultrows = []
    for halfwins in range(int(2*remaining+1)):
        spec_resmat = add_record(resmat, playerA, playerB, halfwins/2, remaining - halfwins/2)
        winsA = repr(spec_resmat.loc[playerA, playerB])
        winsB = repr(spec_resmat.loc[playerB, playerA])
        demoters = could_demote(spec_resmat, ngames, nplayers)
        resultrows.append((winsA, winsB, repr(demoters)))

    return playerA, playerB, resultrows

def impl_text(playerA, playerB, resultrows):
    for winsA, winsB, demoters in resultrows:
        print(f'{playerA} {winsA}-{winsB} {playerB} {demoters}')

s46a1 = """
Season 46 Division A1
Match Summary (current)
Completed matches
nasmith99 6–0 jonts
nasmith99 3–3 crabcat2
nasmith99 4–2 JNails
nasmith99 5–1 recycle_garbage
nasmith99 3–3 aku chi
nasmith99 5–1 tracer
jonts 4–2 crabcat2
jonts 2–4 JNails
jonts 5–1 recycle_garbage
jonts 5–1 aku chi
jonts 3–3 tracer
crabcat2 4–2 JNails
crabcat2 3–3 recycle_garbage
crabcat2 4–2 aku chi
crabcat2 2–4 tracer
JNails 3–3 recycle_garbage
JNails 3–3 aku chi
JNails 4–2 tracer
recycle_garbage 4–2 tracer
aku chi 3–3 tracer
"""
# Removed match for testing:
#recycle_garbage 3–3 aku chi

mat = file_read.from_DomBot_matches(s46a1.splitlines())

#print(mat)
#print(winloss(mat))
#print(promoting(mat))
#print(demoting(mat, nplayers=2))
#print(could_promote(mat, nplayers=1))
#print(could_demote(mat, nplayers=2))

impl_image(*match_impl(mat, 'aku chi', 'recycle_garbage', nplayers=2))

