// Poker table rendering for play mode with animations

const Table = {
    gameId: null,
    agentStrategy: null,
    wins: 0,
    losses: 0,
    lastCardState: null,

    init() {
        document.getElementById('play-new-game').addEventListener('click', () => this.newGame());
        document.getElementById('play-new-hand').addEventListener('click', () => this.newHand());
    },

    async newGame() {
        const status = document.getElementById('play-status');
        status.textContent = 'starting...';
        this.wins = 0;
        this.losses = 0;
        this._updateScore();

        try {
            const result = await API.newGame('kuhn');
            this.gameId = result.game_id;
            this.agentStrategy = null;
            this.lastCardState = null;
            this._clearResult();
            this.renderState(result.state, true);
            document.getElementById('play-new-hand').disabled = false;
            status.textContent = `game ${this.gameId}`;
        } catch (e) {
            status.textContent = `error: ${e.message}`;
        }
    },

    async newHand() {
        if (!this.gameId) return;
        this.lastCardState = null;
        this._clearResult();
        const result = await API.newHand(this.gameId);
        this.renderState(result.state, true);
    },

    async takeAction(action) {
        if (!this.gameId) return;
        const result = await API.takeAction(this.gameId, action);
        this.renderState(result.state, false);

        if (result.terminal) {
            this.agentStrategy = result.agent_strategy;
            this._showResult(result.state);
            // Reveal opponent card
            setTimeout(() => {
                Cards.revealAll(document.getElementById('opponent-cards'));
            }, 200);
            Replay.loadFromGame(this.gameId, result.replay, result.agent_strategy);
        }
    },

    renderState(state, isNewHand = false) {
        const p0Card = state.players[0]?.card;
        const p1Card = state.players[1]?.card;

        // Human cards — always face up
        Cards.renderToContainer(
            document.getElementById('human-cards'),
            p0Card ? [p0Card] : [],
            isNewHand,
        );

        // Opponent cards — face down during play, revealed at terminal
        const oppContainer = document.getElementById('opponent-cards');
        if (isNewHand || !this.lastCardState) {
            // Deal new cards. During play, opponent is face down.
            if (state.is_terminal) {
                Cards.renderToContainer(oppContainer, p1Card ? [p1Card] : [], isNewHand);
            } else {
                Cards.renderToContainer(oppContainer, p1Card ? [p1Card] : [], isNewHand);
            }
        }
        this.lastCardState = { p0Card, p1Card };

        // Pot
        const potDisplay = document.getElementById('pot-display');
        const potAmount = potDisplay.querySelector('.pot-amount');
        const oldPot = parseInt(potAmount.textContent) || 0;
        potAmount.textContent = state.pot;
        if (state.pot !== oldPot && !isNewHand) {
            potDisplay.classList.remove('bump');
            void potDisplay.offsetWidth; // reflow
            potDisplay.classList.add('bump');
        }

        // Action log as pills
        const log = document.getElementById('action-log');
        log.innerHTML = '';
        if (state.action_history.length === 0) {
            log.innerHTML = '<span class="action-pill-wait">your move</span>';
        } else {
            state.action_history.forEach(a => {
                const pill = document.createElement('span');
                pill.className = `action-pill ${a.action}`;
                pill.textContent = `P${a.player + 1} ${a.action}`;
                log.appendChild(pill);
            });
        }

        // Action buttons
        const btnContainer = document.getElementById('action-buttons');
        btnContainer.innerHTML = '';

        if (!state.is_terminal && state.current_player === 0) {
            state.legal_actions.forEach(action => {
                const btn = document.createElement('button');
                btn.className = `action-btn ${action}`;
                btn.textContent = action.charAt(0).toUpperCase() + action.slice(1);
                btn.addEventListener('click', () => this.takeAction(action));
                btnContainer.appendChild(btn);
            });
        }

        // Agent thinking
        const stratDisplay = document.getElementById('agent-strategy-display');
        if (!state.is_terminal && state.current_player === 1) {
            stratDisplay.textContent = 'thinking...';
        } else {
            stratDisplay.textContent = '';
        }
    },

    _showResult(state) {
        const display = document.getElementById('result-display');
        if (state.winner === 0) {
            display.textContent = 'You Win';
            display.className = 'result-overlay win';
            this.wins++;
        } else {
            display.textContent = 'Agent Wins';
            display.className = 'result-overlay lose';
            this.losses++;
        }
        this._updateScore();
    },

    _clearResult() {
        const display = document.getElementById('result-display');
        display.textContent = '';
        display.className = 'result-overlay';
    },

    _updateScore() {
        document.getElementById('score-wins').textContent = this.wins;
        document.getElementById('score-losses').textContent = this.losses;
    },
};
