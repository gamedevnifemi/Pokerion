// Hand replay with step-through, timeline, and history grouped by match.
//
// Two levels: a match (one "New Match") contains many hands. Selecting a match
// shows its hands; selecting a hand steps through it. Backed by the server's
// store, so history survives page reloads and server restarts.

const Replay = {
    sessions: [],          // [{id, label, hands: [...]}]
    activeSessionIdx: -1,
    activeHandIdx: -1,

    // The hand currently loaded into the replay table
    states: [],
    strategy: {},
    currentStep: 0,

    init() {
        document.getElementById('replay-prev').addEventListener('click', () => this.prev());
        document.getElementById('replay-next').addEventListener('click', () => this.next());
    },

    // Open a new match group. Previous matches stay browsable.
    startSession(matchId) {
        this.sessions.push({
            id: matchId,
            label: `M${this.sessions.length + 1}`,
            hands: [],
        });
        this.activeSessionIdx = this.sessions.length - 1;
        this.activeHandIdx = -1;

        this._clearHand();
        this._renderSessions();
        this._renderHands();
        this._updateControls();
        this._clearTable();
    },

    addHand(matchId, states, strategy, meta) {
        const session = this._sessionById(matchId) || this._openSessionFor(matchId);
        session.hands.push(this._makeHand(states, strategy, session.hands.length, meta));

        this.activeSessionIdx = this.sessions.indexOf(session);
        this._selectHand(session.hands.length - 1);
    },

    // Rebuild history from the server after a page reload.
    async restore() {
        try {
            const data = await API.listSessions();
            if (!data.sessions?.length) return;

            this.sessions = data.sessions.map((m, i) => ({
                id: m.id,
                label: `M${i + 1}`,
                hands: (m.hands || []).map((h, j) =>
                    this._makeHand(h.states, h.strategy, j, h)),
            }));

            this.activeSessionIdx = this.sessions.length - 1;
            const session = this._activeSession();
            if (session && session.hands.length) {
                this._selectHand(session.hands.length - 1);
            } else {
                this.activeHandIdx = -1;
                this._renderSessions();
                this._renderHands();
                this._updateControls();
            }
        } catch (e) {
            console.warn('[Replay] could not restore history:', e);
        }
    },

    _makeHand(states, strategy, index, meta = {}) {
        const terminal = states[states.length - 1];
        return {
            id: `${index + 1}`,
            states,
            strategy: strategy || {},
            p0Card: terminal?.players?.[0]?.card || '?',
            p1Card: terminal?.players?.[1]?.card || '?',
            winnerSeat: meta.winner_seat ?? terminal?.winner,
            // No `?? 0` default. Assuming seat 0 silently inverts the W/L dot
            // on every rotated hand; null makes a missing value visible as
            // "unknown" instead of confidently wrong.
            humanSeat: meta.human_seat ?? null,
            chipsDelta: meta.chips_delta,
            actions: terminal?.action_history?.map(a => a.action).join('-') || '',
        };
    },

    _humanWon(hand) {
        if (hand.humanSeat === null || hand.winnerSeat === null) return null;
        return hand.winnerSeat === hand.humanSeat;
    },

    // Full wipe — not used by the normal flow, kept for a hard reset.
    reset() {
        this.sessions = [];
        this.activeSessionIdx = -1;
        this.activeHandIdx = -1;

        this._clearHand();
        this._renderSessions();
        this._renderHands();
        this._updateControls();
        this._clearTable();
    },

    // --- internals -------------------------------------------------------

    _activeSession() {
        return this.sessions[this.activeSessionIdx] || null;
    },

    _sessionById(matchId) {
        return this.sessions.find(s => s.id === matchId) || null;
    },

    // Safety net: a hand arriving for a match we never opened.
    _openSessionFor(matchId) {
        this.startSession(matchId);
        return this.sessions[this.sessions.length - 1];
    },

    _clearHand() {
        this.states = [];
        this.strategy = {};
        this.currentStep = 0;
    },

    _selectSession(idx) {
        if (idx < 0 || idx >= this.sessions.length) return;

        this.activeSessionIdx = idx;
        const session = this.sessions[idx];

        if (session.hands.length === 0) {
            this.activeHandIdx = -1;
            this._clearHand();
            this._renderSessions();
            this._renderHands();
            this._updateControls();
            this._clearTable();
            return;
        }
        this._selectHand(session.hands.length - 1);
    },

    _selectHand(idx) {
        const session = this._activeSession();
        if (!session || idx < 0 || idx >= session.hands.length) return;

        this.activeHandIdx = idx;
        const hand = session.hands[idx];
        this.states = hand.states;
        this.strategy = hand.strategy;
        this.currentStep = 0;

        this._renderSessions();
        this._renderHands();
        this._updateControls();
        this.renderStep(false);
    },

    _renderSessions() {
        const container = document.getElementById('session-history');
        container.innerHTML = '';

        if (this.sessions.length === 0) {
            container.innerHTML = '<span class="history-empty">No matches yet</span>';
            return;
        }

        this.sessions.forEach((session, i) => {
            const judged = session.hands.filter(h => this._humanWon(h) !== null);
            const wins = judged.filter(h => this._humanWon(h)).length;
            const losses = judged.length - wins;

            const chip = document.createElement('button');
            chip.className = 'session-chip';
            if (i === this.activeSessionIdx) chip.classList.add('active');
            chip.innerHTML =
                `<span class="session-name">${session.label}</span>` +
                `<span class="session-record">${wins}-${losses}</span>`;
            chip.addEventListener('click', () => this._selectSession(i));
            container.appendChild(chip);
        });
    },

    _renderHands() {
        const container = document.getElementById('hand-history');
        container.innerHTML = '';

        const session = this._activeSession();
        if (!session) {
            container.innerHTML = '<span class="history-empty">Play some hands first</span>';
            return;
        }
        if (session.hands.length === 0) {
            container.innerHTML = '<span class="history-empty">No hands in this match yet</span>';
            return;
        }

        session.hands.forEach((hand, i) => {
            const chip = document.createElement('button');
            chip.className = 'hand-chip';
            if (i === this.activeHandIdx) chip.classList.add('active');

            const won = this._humanWon(hand);
            const dot = document.createElement('span');
            dot.className = `hand-result ${won === null ? 'u' : won ? 'w' : 'l'}`;

            const label = document.createElement('span');
            label.textContent = `#${hand.id} ${hand.p0Card}v${hand.p1Card}`;

            chip.appendChild(dot);
            chip.appendChild(label);
            chip.addEventListener('click', () => this._selectHand(i));
            container.appendChild(chip);
        });
    },

    _clearTable() {
        document.getElementById('replay-p1-cards').innerHTML = '';
        document.getElementById('replay-p2-cards').innerHTML = '';
        document.getElementById('replay-pot').querySelector('.pot-amount').textContent = '0';
        document.getElementById('replay-action-log').innerHTML = '';
        document.getElementById('replay-strategy').textContent = '';

        const resultEl = document.getElementById('replay-result');
        resultEl.textContent = '';
        resultEl.className = 'result-overlay';
    },

    // --- step-through ----------------------------------------------------

    prev() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this._updateControls();
            this.renderStep(true);
        }
    },

    next() {
        if (this.currentStep < this.states.length - 1) {
            this.currentStep++;
            this._updateControls();
            this.renderStep(true);
        }
    },

    renderStep(animate = true) {
        if (this.states.length === 0) return;

        const state = this.states[this.currentStep];
        const isFirst = this.currentStep === 0;

        // Cards (god mode) — replay keeps engine-seat orientation: seat 0
        // bottom, seat 1 top. Consistent across hands even as the human's
        // seat rotates.
        const p0Card = state.players[0]?.card;
        const p1Card = state.players[1]?.card;

        Cards.renderToContainer(
            document.getElementById('replay-p1-cards'),
            p0Card ? [p0Card] : [],
            isFirst && animate,
        );

        Cards.renderToContainer(
            document.getElementById('replay-p2-cards'),
            p1Card ? [p1Card] : [],
            isFirst && animate && !state.is_terminal,
        );

        // Pot
        document.getElementById('replay-pot').querySelector('.pot-amount').textContent = state.pot;

        // Action log
        const log = document.getElementById('replay-action-log');
        log.innerHTML = '';
        if (state.action_history.length === 0) {
            log.innerHTML = '<span class="action-pill-wait">deal</span>';
        } else {
            state.action_history.forEach(a => {
                const pill = document.createElement('span');
                pill.className = `action-pill ${a.action}`;
                pill.textContent = `P${a.player + 1} ${a.action}`;
                log.appendChild(pill);
            });
        }

        // Strategy info panel
        const stratEl = document.getElementById('replay-strategy');
        const resultEl = document.getElementById('replay-result');
        resultEl.textContent = '';
        resultEl.className = 'result-overlay';

        if (!state.is_terminal && !state.is_chance && state.current_player !== null) {
            const player = state.current_player;
            const card = state.players[player]?.card;
            const actionStr = state.action_history.map(a => a.action).join(':');
            const key = actionStr ? `${card}|${actionStr}` : card;

            if (this.strategy[key]) {
                const strat = this.strategy[key];
                const parts = Object.entries(strat)
                    .map(([a, p]) => `${a}: ${(p * 100).toFixed(1)}%`)
                    .join('  /  ');
                stratEl.innerHTML = `<strong>P${player + 1}</strong> at <code>${key}</code> &mdash; ${parts}`;
            } else {
                stratEl.textContent = `P${player + 1} to act`;
            }
        } else if (state.is_terminal) {
            stratEl.textContent = 'Hand complete';
            if (state.winner !== null) {
                const hand = this._activeSession()?.hands[this.activeHandIdx];
                const humanWon = hand ? this._humanWon(hand) : null;
                resultEl.textContent = `Player ${state.winner + 1} wins`;
                resultEl.className = 'result-overlay'
                    + (humanWon === null ? '' : humanWon ? ' win' : ' lose');
            }
        } else {
            stratEl.textContent = '';
        }
    },

    _updateControls() {
        const total = this.states.length;
        document.getElementById('replay-prev').disabled = this.currentStep <= 0;
        document.getElementById('replay-next').disabled = this.currentStep >= total - 1;
        document.getElementById('replay-step').textContent =
            total > 0 ? `Step ${this.currentStep + 1} / ${total}` : 'No replay loaded';

        const fill = document.getElementById('replay-fill');
        if (total > 1) {
            fill.style.width = `${(this.currentStep / (total - 1)) * 100}%`;
        } else {
            fill.style.width = '0%';
        }
    },
};
