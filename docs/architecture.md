# Architecture

Two things live in this repository, and they share the game definitions.

- **The site** (pokerion.dev): train a Kuhn poker agent, play against it, replay the hands.
- **The lab**: exact tools for studying strategies in small poker games, for any number of players. It is where new methods are built and measured before anything goes on the site.

## Map

```
src/pokerion/
  game/
    base.py           History: the interface every game implements
    kuhn.py           Kuhn poker for two or more players
    tree.py           GameTree: a game compiled into arrays              lab
  evaluation/         best response, deviation gains, NashConv           lab
  solver/
    tree_cfr.py       CFR on a GameTree                                  lab
    regret_models.py  where that solver keeps its regrets                lab
    vanilla_cfr.py    CFR by recursion over a History                    site
    info_set.py       one information set's regrets, for vanilla_cfr     site
  arena/              simple bots, and exact scoring of a table of bots  lab
  training/           runs the recursive solver with progress snapshots  site
  server/             API, training jobs, storage, play against a human  site
frontend/             the web page
tests/
docs/
```

Dependencies point one way. `game` depends on nothing. `evaluation`, `solver` and `arena` depend on `game`. `server` depends on `training`, `solver` and `game`. Nothing in the lab imports from `server`.

## The layers of the lab

**A game is a `History`.** `game/base.py` defines a node of the game tree: who acts, which actions are legal, what the acting player can see (the information set key), and the payoffs at the end. `KuhnHistory` implements it. This form is easy to write and easy to read, and slow to compute with.

**A `GameTree` is that game compiled into arrays.** `GameTree(root)` walks the game once and stores every node and edge in numpy arrays, in breadth-first order. After that, the quantities everything else needs are a few array operations each:

- `reach(profile)`: how likely each node is, split into one factor per player and one for chance;
- `expected_utilities(profile)`: what each seat wins per hand;
- `node_values(profile)`: what each seat expects from every node onward.

It only suits games small enough to enumerate. The module docstring of `game/tree.py` describes the layout.

**`evaluation` asks how good a profile is.** `best_response` finds the most one seat can win against the others. `deviation_gains` is how much each seat would gain by switching to it, and `nash_conv` is their sum. All are exact.

**`solver/tree_cfr.py` finds solid play.** `TreeCFR` runs counterfactual regret minimisation on a `GameTree`. It keeps its regrets in a `RegretModel`, so a compact model can replace the table without the solver changing. `RegretTable` is the exact reference, with a `floored` option that gives regret matching+.

**`arena` seats strategies at a table.** `arena/bots.py` has simple fixed strategies: random, calling station, maniac, rock. `arena/table.py` computes what each entrant wins per hand, averaged over every seating, and the most that a given seat could have won.

## Conventions

- **Seats** are numbered from 0. Seat 0 acts first.
- **Payoffs** are chips per hand, from the point of view of the seat named.
- **An information set** is everything a player knows when it acts: its own card and the betting so far. Its key is a string such as `"K|check:bet"`.
- **A strategy has two forms.**
  - A dictionary, `{key: {action: probability}}`. This is what the site stores and what the recursive solver produces.
  - An array of shape `(num_infosets, num_actions)`, one row per information set and one column per action, with zeros for actions that are not legal in that row. This is what the lab computes with.
  - `GameTree.strategy_from_dict` and `GameTree.strategy_to_dict` convert between them.
- **A profile** is a strategy for every seat. Information sets of different seats are different rows, so a profile has the same shape as one seat's strategy. `GameTree.combine` builds a profile from one array per seat.
- **NashConv and exploitability.** NashConv is the sum over seats of what each would gain by best-responding. The site's training view shows NashConv and calls it exploitability. Papers on two-player games usually call half of NashConv the exploitability.

## Using the lab

```python
from pokerion.arena import Entrant, bots, evaluate_table
from pokerion.evaluation import nash_conv
from pokerion.game.kuhn import KUHN_3P
from pokerion.game.tree import GameTree
from pokerion.solver.regret_models import RegretTable
from pokerion.solver.tree_cfr import TreeCFR

tree = GameTree(KUHN_3P.root())

# Solid play: CFR+ for 1,500 iterations.
solver = TreeCFR(tree, RegretTable(tree, floored=True), linear_averaging=True)
solver.train(1500)
solid = solver.average_strategy()

print(tree.expected_utilities(solid))   # about [-0.027, -0.021, 0.047]
print(nash_conv(tree, solid))           # close to 0

# What does solid play win at a table with a rock and a maniac?
table = [
    Entrant("solid", solid),
    Entrant("rock", bots.rock(tree)),
    Entrant("maniac", bots.maniac(tree)),
]
print(evaluate_table(tree, table).winnings)   # about [0.16, -1.0, 0.84]
```

## Checked against known results

Nothing in the lab counts until it reproduces numbers that are already known. These are tested on every run.

| Game | Quantity | Known value | Source | Test |
|---|---|---|---|---|
| Kuhn, 2 players | First player's equilibrium value | -1/18 | Kuhn 1950 | `test_tree_cfr.py` |
| Kuhn, 2 players | NashConv of uniform play | 11/12 | OpenSpiel's exploitability tests | `test_evaluation.py` |
| Kuhn, 2 players | Equilibrium properties, such as bluffing a jack one third of the time | | Kuhn 1950 | `test_tree_cfr.py` |
| Kuhn, 3 players | Number of information sets | 48 | Szafron, Gibson and Sturtevant 2013 | `test_kuhn_multiplayer.py` |
| Kuhn, 3 players | The paper's worked example of a hand | payoffs -2, 3, -1 | same, section 2 | `test_kuhn_multiplayer.py` |
| Kuhn, 3 players | Second player's equilibrium payoff | -1/48 | same, Table 3 | `test_tree_cfr.py` |
| Kuhn, 3 players | First and third players' equilibrium payoffs | between -1/32 and -1/48, and between 1/24 and 5/96 | same, Table 3 | `test_tree_cfr.py` |

The array engine is also checked against calculations that share no code with it: a plain recursive walk of the game for expected payoffs, trying every pure strategy for best responses, and the recursive solver's own best response.

## Extending it

**A new game.** Implement `History`. `GameTree` checks the rest when it builds:

- chance probabilities at a node must sum to one;
- an information set key must always belong to the same seat and offer the same legal actions;
- `best_response` also needs every node of an information set to sit at the same depth, which holds whenever all betting is public.

Use the action names "check", "bet", "call", "raise" and "fold" if the bots in `arena/bots.py` should work in the game.

**A new way of storing regrets.** Implement the two methods of `RegretModel` in `solver/regret_models.py` and pass an instance to `TreeCFR`. Compare it with `RegretTable` at the same number of iterations.

**A new bot.** Add a function to `arena/bots.py` that takes a `GameTree` and returns a strategy array.

## Two solvers, for now

`VanillaCFR` is the solver the site runs. `TreeCFR` is the solver the lab uses. They are separate on purpose in this change, so that the site's behaviour does not move while the new engine is being proven. The intention is to move the site onto the tree engine and remove the recursive solver.

Until then, two differences matter:

- **`VanillaCFR` is only correct when chance is uniform.** Its regret update does not weight a node by its chance probability. In Kuhn every deal is equally likely, so the weight is a constant and the result is right. In a game with uneven chance it is wrong: on a coin that lands heads 70% of the time it learns to guess at random. `tests/test_tree_cfr.py` holds this as an expected failure. `TreeCFR` weights by chance and is tested on the same game.
- **`VanillaCFR` refreshes its strategy during an iteration.** An information set is visited once per deal, and each visit uses regrets already updated by the previous one. `TreeCFR` fixes the strategy for the whole pass, as the algorithm is usually stated. Both converge on Kuhn, by different paths, so their strategies agree only in the limit.

## Not built yet

- Matches between agents that change how they play from hand to hand. `arena` scores fixed strategies only.
- A memory of how each opponent has played, and a rule for using it safely.
- Leduc poker, and games with a size knob.
- Regret models other than the table.
- Solving from sampled hands instead of the whole tree.

## Working on it

```
pip install -e ".[dev]"
pytest -q
python -m pokerion        # the site, on http://127.0.0.1:8000
```

CI runs the tests on Python 3.13 and 3.14. A push to `main` builds and deploys the site, so work on a branch.
