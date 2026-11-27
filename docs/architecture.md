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
  arena/
    bots.py           simple fixed strategies to play against            lab
    table.py          fixed strategies, scored over every seating        lab
    match.py          many hands between agents, scored hand by hand     lab
    agents.py         an agent that adapts: core + memory + rule         lab
    memory.py         what an agent remembers about each opponent        lab
    exploitation.py   how that memory changes what the agent plays       lab
  training/           runs the recursive solver with progress snapshots  site
  server/             API, training jobs, storage, play against a human  site
    routes/lab.py     plays a lab match for the Lab page; local use only lab
frontend/             the web page, including the Lab tab
experiments/          studies that take minutes and print a table       lab
tests/
docs/
```

Dependencies point one way. `game` depends on nothing. `evaluation`, `solver` and `arena` depend on `game`. `server` depends on `training`, `solver` and `game`, and its Lab route depends on `arena`. Nothing in the lab imports from `server`.

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

**`arena/match.py` plays matches between agents that adapt.** Before each hand every agent says which strategy it will play, and the hand is scored exactly from the tree. One hand is also dealt at random, and each agent is shown what a player in its seat would have seen of it. So luck decides only what the agents learn from, never who is credited with the pot.

**An adapting agent has three parts**, each of which can be swapped on its own:

| Part | What it does | Interface | Built so far |
|---|---|---|---|
| Core | Solid play, used when nothing is known about the table | a strategy array | from `TreeCFR` |
| Memory | Predicts each opponent's strategy from the hands seen | `OpponentMemory` | `CountingMemory` |
| Rule | Turns the predictions into the strategy to play | `ExploitationRule` | `PlayTheCore`, `MixWithBestResponse` |

`AdaptiveAgent` puts them together. With the rule `PlayTheCore` it never adapts, which makes it the reference that adapting versions are measured against.

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
- **Agents and seats.** In a match an agent keeps one number for the whole match: its position in the list of agents. The seats change from hand to hand, and a seating says which agent is in each seat. By default a match cycles through every seating.
- **What an agent is shown.** Every action is public. The information set behind an action is shown only when its cards are: an agent's own always, and another player's when the hand ends in a showdown that player is part of. That is `reveal="showdown"`, the default. `reveal="all"` turns every card over after every hand, as the published Kuhn experiments do.

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

The same table, with an agent that adapts in place of fixed solid play:

```python
import numpy as np

from pokerion.arena import AdaptiveAgent, CountingMemory, MixWithBestResponse, play_match

hero = AdaptiveAgent("hero", solid, CountingMemory(prior=solid), MixWithBestResponse(1.0))
match = play_match(
    tree, [hero, table[1], table[2]], hands=300, rng=np.random.default_rng(0), reveal="all"
)
won = match.per_hand[0]                       # what the hero expected to win on each hand
print(won[:30].mean(), won[-30:].mean())      # about 0.49 early and 0.64 late: it learns
```

## Watching the lab in the browser

```
python -m pokerion
```

Then open http://127.0.0.1:8000 and choose the **Lab** tab. Pick a game, choose who sits at the table, and press "Play match". The page shows one match, top to bottom:

- **The score.** Each player's chips per hand through the match, as a moving average. Every hand is scored by what each player expected to win, so the lines carry no card luck. For an adapting agent facing fixed bots, two dashed lines mark what it would have won without adapting and the most anything could have won in its place. The gap between them is what there is to gain.
- **The hand that was dealt.** One hand is dealt each time for the agents to learn from. Step through them to see the seats, the cards, the betting and which cards the others got to see.
- **What the agent believed.** What the adapting agent expected each opponent to do, at that point in the match, beside what the opponent really does.

Two settings on an adapting agent are worth trying, because each shows a known limit of the first agent:

- "Cards shown after a hand", set to only cards in a showdown. Against a rock the agent stops improving, because a fold never shows a card.
- "Equal choices go to", set to checking and folding. Against a random bot the agent ends up far lower, because it never tries the actions it would have learned from.

**The Lab is for local use.** Its routes answer 404 unless the environment variable `POKERION_LAB` is set. `python -m pokerion` sets it. The deployed site is started by its container, which does not, so the Lab tab stays hidden there. The reason is that one request plays a whole match and nothing limits how often it is asked. Opening it to the public would mean putting it behind the same job queue and limits as training.

The page is `frontend/js/components/lab.js`, and it gets everything from two routes in `server/routes/lab.py`: `GET /api/lab/options` for the choices, and `POST /api/lab/match` for a match. A match is capped at 2,000 hands.

## Checked against known results

Nothing in the lab counts until it reproduces numbers that are already known. The ones in this table are tested on every run.

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

One check takes minutes, so it is an experiment and not a test. `experiments/safe_exploitation_kuhn.py` replays the "best response" row of Table I in Ganzfried and Sandholm, "Safe Opponent Exploitation" (2015): an agent that counts its opponent's actions and best-responds to the counts, as first player in two-player Kuhn, for 1,000 hands against each of 300 opponents. Results from 2026-11-27, in chips per hand with 95% intervals:

| Opponents | Published | Near-ties to bets and calls | Near-ties to checks and folds |
|---|---|---|---|
| Random | 0.4700 | 0.493 ± 0.037 | 0.268 ± 0.024 |
| Sophisticated static | 0.0548 | 0.046 ± 0.006 | 0.029 ± 0.005 |
| Equilibrium | -0.0556 | -0.0556 | -0.0556 |

Read it this way. Against random opponents and against the equilibrium the published figures are reproduced, when near-ties go to bets and calls. Against the sophisticated static opponents the result is close but about a sixth lower than published, and the published figure is just outside the interval; the cause has not been found. The paper does not say how its best response breaks ties, and the last column shows how much that choice moves the result.

## Extending it

**A new game.** Implement `History`. `GameTree` checks the rest when it builds:

- chance probabilities at a node must sum to one;
- an information set key must always belong to the same seat and offer the same legal actions;
- `best_response` also needs every node of an information set to sit at the same depth, which holds whenever all betting is public.

Use the action names "check", "bet", "call", "raise" and "fold" if the bots in `arena/bots.py` should work in the game.

**A new way of storing regrets.** Implement the two methods of `RegretModel` in `solver/regret_models.py` and pass an instance to `TreeCFR`. Compare it with `RegretTable` at the same number of iterations.

**A new bot.** Add a function to `arena/bots.py` that takes a `GameTree` and returns a strategy array.

**A new memory.** Implement the three methods of `OpponentMemory` in `arena/memory.py`: start a match, take in a finished hand, predict an opponent's strategy. Compare it with `CountingMemory`, keeping the core and the rule the same.

**A new rule.** Implement `choose` from `ExploitationRule` in `arena/exploitation.py`. Compare it with `PlayTheCore` and `MixWithBestResponse`, keeping the core and the memory the same.

**A new kind of agent.** Anything with `begin_match`, `strategy_for` and `observe` can sit in a match. See `Agent` in `arena/match.py`.

## Two solvers, for now

`VanillaCFR` is the solver the site runs. `TreeCFR` is the solver the lab uses. They are separate on purpose in this change, so that the site's behaviour does not move while the new engine is being proven. The intention is to move the site onto the tree engine and remove the recursive solver.

Until then, two differences matter:

- **`VanillaCFR` is only correct when chance is uniform.** Its regret update does not weight a node by its chance probability. In Kuhn every deal is equally likely, so the weight is a constant and the result is right. In a game with uneven chance it is wrong: on a coin that lands heads 70% of the time it learns to guess at random. `tests/test_tree_cfr.py` holds this as an expected failure. `TreeCFR` weights by chance and is tested on the same game.
- **`VanillaCFR` refreshes its strategy during an iteration.** An information set is visited once per deal, and each visit uses regrets already updated by the previous one. `TreeCFR` fixes the strategy for the whole pass, as the algorithm is usually stated. Both converge on Kuhn, by different paths, so their strategies agree only in the limit.

## What the first adapting agent can and cannot do

The first memory counts actions and the first rule best-responds to the counts. They are the simplest contenders, and two of their limits are already visible.

- **Counting is blind to hands with no showdown.** A fold never shows the folded cards, so under `reveal="showdown"` the agent never learns that a rock folds. `tests/test_arena_agents.py` holds this as a test.
- **Best-responding to counts never tests a belief.** Where the agent has no data it keeps trusting its prior, and if the prior says an action is bad it never tries the action, so it never gets the data. Against a bot that plays at random it settles well below what is available.

Neither limit is a bug in the match play. They are what the next memories and rules have to beat.

## Not built yet

- A memory that learns from hands with no showdown.
- Rules with a stated bound on risk, and a rule that explores.
- Opponents that learn, and adversaries that are allowed to see our strategy.
- Leduc poker, and games with a size knob.
- Regret models other than the table.
- Solving from sampled hands instead of the whole tree.

## Working on it

```
pip install -e ".[dev]"
pytest -q
python -m pokerion                              # the site, on http://127.0.0.1:8000
python -m experiments.safe_exploitation_kuhn    # a study; each one says how long it takes
```

Each experiment has a test in `tests/test_experiments.py` that runs it at a tiny size, so that it keeps working as the code changes.

CI runs the tests on Python 3.13 and 3.14. A push to `main` builds and deploys the site, so work on a branch.
