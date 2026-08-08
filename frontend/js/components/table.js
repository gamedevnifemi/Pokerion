// Poker table rendering for match play.
//
// Orientation comes from state.match.human_seat — the human is whichever
// engine seat the match says, never players[0] by assumption. Seats rotate
// every hand because Kuhn's opening seat loses 1/18 per hand at equilibrium;
// alternation cancels the structural bias over an even match.

const NASH_NOTE = 'Nash expectation with alternating seats: 0.00 chips/hand';

const Table = {
    matchId: null,
    match: null,        // latest match meta from the server
    wins: 0,
    losses: 0,

    init() {
        document.getElementById('play-new-match').addEventListener('click', () => this.newMatch());
        document.getElementById('play-new-hand').addEventListener('click', () => this.newHand());
    },

    // Reattach to an unfinished match after a page reload.
    //
    // Replay restored itself but Table did not, so a refresh mid-match left the
    // player unable to act or Deal while the server-side match sat there for
    // two hours — and the end-of-match summary could never fire.
    async restore() {
        try {
            const data = await API.listSessions();
            const live = (data.sessions || []).filter(m => !m.complete).pop();
            if (!live) return;

            const result = await API.getMatchState(live.id);
            if (result.error) return;

            this.matchId = live.id;
            this.match = result.state.match;
            // Rebuild the hand tally from stored records rather than assuming zero.
            const hands = live.hands || [];
            this.wins = hands.filter(h => h.winner_seat === h.human_seat).length;
            this.losses = hands.length - this.wins;
            this._updateScore();

            this.renderState(result.state, true);
            document.getElementById('play-new-hand').disabled = false;
            document.getElementById('play-status').textContent = `match ${this.matchId} (resumed)`;
        } catch (e) {
            console.warn('[Table] could not resume match:', e);
        }
    },

    async newMatch() {
        const status = document.getElementById('play-status');
        const length = parseInt(document.getElementById('match-length').value) || 50;
        status.textContent = 'starting...';
        this.wins = 0;
        this.losses = 0;
        this._updateScore();
        this._clearResult();
        this._hideSummary();

        try {
            const result = await API.newMatch('kuhn', length);
            if (result.error) { status.textContent = `error: ${result.error}`; return; }
            this.matchId = result.match_id;
            this.match = result.state.match;
            Replay.startSession(this.matchId);
            this.renderState(result.state, true);
            document.getElementById('play-new-hand').disabled = false;
            status.textContent = `match ${this.matchId}`;
        } catch (e) {
            status.textContent = `error: ${e.message}`;
        }
    },

    async newHand() {
        if (!this.matchId) return;
        this._clearResult();
        const result = await API.newHand(this.matchId);
        if (result.error) {
            document.getElementById('play-status').textContent = result.error;
            return;
        }
        this.match = result.state.match;
        this.renderState(result.state, true);
    },

    async takeAction(action) {
        if (!this.matchId) return;
        const result = await API.takeAction(this.matchId, action);
        if (result.error) {
            document.getElementById('play-status').textContent = result.error;
            return;
        }
        this.match = result.state.match;
        this.renderState(result.state, false);

        if (result.terminal) {
            const meta = result.hand_meta;
            this._showHandResult(meta);
            setTimeout(() => {
                Cards.revealAll(document.getElementById('opponent-cards'));
            }, 200);
            Replay.addHand(this.matchId, result.replay, result.agent_strategy, meta);

            if (this.match.complete) {
                document.getElementById('play-new-hand').disabled = true;
                await this._showMatchSummary();
            }
        }
    },

    renderState(state, isNewHand = false) {
        const humanSeat = state.match.human_seat;
        const agentSeat = 1 - humanSeat;
        const humanCard = state.players[humanSeat]?.card;
        const agentCard = state.players[agentSeat]?.card;

        // Human cards — bottom, always face up
        Cards.renderToContainer(
            document.getElementById('human-cards'),
            humanCard ? [humanCard] : [],
            isNewHand,
        );

        // Agent cards — top, face down during play, revealed at terminal
        Cards.renderToContainer(
            document.getElementById('opponent-cards'),
            agentCard ? [agentCard] : [],
            isNewHand,
        );

        // Chips: cumulative signed P&L, zero-sum — the agent's is the mirror.
        const chips = state.match.human_chips;
        this._setChips('human-chips', chips);
        this._setChips('agent-chips', -chips);

        // Match progress + seat indicator
        const handNo = state.is_terminal ? state.match.hand_index : state.match.hand_index + 1;
        document.getElementById('match-progress').textContent =
            `Hand ${Math.min(handNo, state.match.length)} / ${state.match.length}`;
        document.getElementById('seat-indicator').textContent =
            humanSeat === 0 ? 'you act first' : 'agent acts first';

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

        // Action log as pills — labelled you/agent, not by engine seat
        const log = document.getElementById('action-log');
        log.innerHTML = '';
        if (state.action_history.length === 0) {
            log.innerHTML = '<span class="action-pill-wait">your move</span>';
        } else {
            state.action_history.forEach(a => {
                const pill = document.createElement('span');
                pill.className = `action-pill ${a.action}`;
                const who = a.player === humanSeat ? 'You' : 'Agent';
                pill.textContent = `${who} ${a.action}`;
                log.appendChild(pill);
            });
        }

        // Action buttons — only when it is actually the human's turn
        const btnContainer = document.getElementById('action-buttons');
        btnContainer.innerHTML = '';
        if (!state.is_terminal && state.current_player === humanSeat) {
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
        stratDisplay.textContent =
            (!state.is_terminal && state.current_player === agentSeat) ? 'thinking...' : '';
    },

    _setChips(elementId, value) {
        const el = document.getElementById(elementId);
        const sign = value > 0 ? '+' : '';
        el.textContent = `${sign}${value}`;
        el.classList.toggle('positive', value > 0);
        el.classList.toggle('negative', value < 0);
    },

    _showHandResult(meta) {
        const display = document.getElementById('result-display');
        const humanWon = meta.winner_seat === meta.human_seat;
        const delta = meta.chips_delta > 0 ? `+${meta.chips_delta}` : `${meta.chips_delta}`;
        display.textContent = humanWon ? `You win ${delta}` : `Agent wins ${delta}`;
        display.className = 'result-overlay ' + (humanWon ? 'win' : 'lose');
        if (humanWon) { this.wins++; } else { this.losses++; }
        this._updateScore();
    },

    async _showMatchSummary() {
        // Fetch deltas from the store rather than trusting client memory —
        // survives a mid-match reload.
        const data = await API.getReplay(this.matchId);
        const deltas = (data.hands || []).map(h => h.chips_delta);
        const n = deltas.length;
        const total = deltas.reduce((a, b) => a + b, 0);
        const mean = n ? total / n : 0;

        // Kuhn's per-hand outcome is a discrete ±1/±2, and its standard
        // deviation (~1.30) dwarfs the 1/18 edge being measured. A normal
        // approximation with z=1.96 is simply wrong at small n — at n=2 it
        // renders "±3.9" where the honest interval is ±25. Use Student-t, and
        // below MIN_CI_HANDS don't pretend to have an interval at all.
        const MIN_CI_HANDS = 20;
        const T_95 = {  // two-tailed 95%, by degrees of freedom (n-1)
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
            20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
        };
        const tCritical = (df) => {
            if (df >= 120) return 1.96;
            const key = Object.keys(T_95).map(Number).filter(k => k >= df)[0];
            return T_95[key] ?? 1.96;
        };

        let ciText;
        if (n >= MIN_CI_HANDS) {
            const sd = Math.sqrt(deltas.reduce((s, d) => s + (d - mean) ** 2, 0) / (n - 1));
            const halfWidth = tCritical(n - 1) * sd / Math.sqrt(n);
            ciText = `${mean.toFixed(3)} ± ${halfWidth.toFixed(3)} chips/hand`;
        } else {
            ciText = `${mean.toFixed(3)} chips/hand (too few hands for an interval)`;
        }

        const verdict = this.match.winner === 'human' ? 'You win the match'
            : this.match.winner === 'agent' ? 'Agent wins the match'
            : 'Draw';

        const wins = (data.hands || []).filter(h => h.winner_seat === h.human_seat).length;

        const panel = document.getElementById('match-summary');
        panel.innerHTML = `
            <h3 class="${this.match.winner === 'human' ? 'win' : this.match.winner === 'agent' ? 'lose' : ''}">${verdict}</h3>
            <div class="summary-grid">
                <div><span class="summary-label">Chips</span><span class="summary-value">${total > 0 ? '+' : ''}${total}</span></div>
                <div><span class="summary-label">Per hand</span><span class="summary-value">${ciText}</span></div>
                <div><span class="summary-label">Hands won</span><span class="summary-value">${wins} of ${n}</span></div>
            </div>
            <p class="summary-note">${NASH_NOTE}. Winning more hands than chips means the agent
            folded often but made you pay at showdown — the core lesson of Kuhn.</p>
            <button class="play-btn primary" id="summary-new-match">New Match</button>
        `;
        panel.classList.add('visible');
        document.getElementById('summary-new-match')
            .addEventListener('click', () => this.newMatch());
    },

    _hideSummary() {
        const panel = document.getElementById('match-summary');
        panel.classList.remove('visible');
        panel.innerHTML = '';
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
