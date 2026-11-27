## Learning/practising Game Theory and RL looking to have lots of fun
### This is a game theory poker lab, here I implement concepts from game theory and reinforcement learning and then and test them using poker.
**Kuhn Poker**:
- Set up Kuhn poker with a Vanilla Counterfactual Regret Minimisation(CFR) solver and trained the Agent using it. 
- Exploitability tracking — measuring how far the learned strategy is from the Nash equilibrium as training progresses.
- There's a web page where I train the agent, play against it and then replay the decisions that went into the hands the agent played and see it's strategy at each decision point.(Pretty straightforward for Kuhn poker, but we'll be moving on to Leduc Hold'em next)

**The lab**:
- Kuhn poker for any number of players, including the standard three-player game.
- An exact engine: the whole game is compiled into arrays, so expected winnings, best responses and distance from equilibrium are computed, not sampled.
- CFR on that engine, with regret matching+ and a slot for other ways of storing regrets.
- A table of bots (random, calling station, maniac, rock), scored exactly over every seating.
- Matches between agents that adapt from hand to hand. An adapting agent is a solid core, a memory of each opponent and a rule for using it, and each part can be swapped.
- All of it is checked against published results for two- and three-player Kuhn.

### Running it
```
pip install -e ".[dev]"
pytest -q
python -m pokerion        # the site, on http://127.0.0.1:8000
```

### Watching the lab
Run the site and open the **Lab** tab. Seat bots at a table, play a match, and see each player's winnings hand by hand, the hands that were dealt, and what an adapting agent believed about its opponents as it went. The Lab is switched on for local runs only.

### How the code is organised
See [docs/architecture.md](docs/architecture.md) for the map of the code, the conventions it follows, a short example of using the lab, and the list of known results it is tested against.
