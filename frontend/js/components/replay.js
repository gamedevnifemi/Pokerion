// Hand replay with step-through, timeline, and hand history

const Replay = {
    states: [],
    strategy: {},
    currentStep: 0,
    gameId: null,

    // Hand history
    hands: [],       // [{id, states, strategy, p0Card, p1Card, winner, actions}]
    activeHandIdx: -1,

    init() {
        document.getElementById('replay-prev').addEventListener('click', () => this.prev());
        document.getElementById('replay-next').addEventListener('click', () => this.next());
    },

    loadFromGame(gameId, states, strategy) {
        // Get terminal state for summary
        const terminal = states[states.length - 1];
        const hand = {
            id: `${this.hands.length + 1}`,
            states,
            strategy: strategy || {},
            p0Card: terminal?.players?.[0]?.card || '?',
            p1Card: terminal?.players?.[1]?.card || '?',
            winner: terminal?.winner,
            actions: terminal?.action_history?.map(a => a.action).join('-') || '',
        };

        this.hands.push(hand);
        this._renderHistory();
        this._selectHand(this.hands.length - 1);
    },

    _selectHand(idx) {
        if (idx < 0 || idx >= this.hands.length) return;

        this.activeHandIdx = idx;
        const hand = this.hands[idx];
        this.gameId = hand.id;
        this.states = hand.states;
        this.strategy = hand.strategy;
        this.currentStep = 0;
        this._updateControls();
        this._renderHistory();
        this.renderStep(false);
    },

    _renderHistory() {
        const container = document.getElementById('hand-history');
        container.innerHTML = '';

        if (this.hands.length === 0) {
            container.innerHTML = '<span class="history-empty">Play some hands first</span>';
            return;
        }

        this.hands.forEach((hand, i) => {
            const chip = document.createElement('button');
            chip.className = 'hand-chip';
            if (i === this.activeHandIdx) chip.classList.add('active');

            const dot = document.createElement('span');
            dot.className = `hand-result ${hand.winner === 0 ? 'w' : 'l'}`;

            const label = document.createElement('span');
            label.textContent = `#${hand.id} ${hand.p0Card}v${hand.p1Card}`;

            chip.appendChild(dot);
            chip.appendChild(label);
            chip.addEventListener('click', () => this._selectHand(i));
            container.appendChild(chip);
        });
    },

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

        // Cards (god mode)
        const p0Card = state.players[0]?.card;
        const p1Card = state.players[1]?.card;

        Cards.renderToContainer(
            document.getElementById('replay-p1-cards'),
            p0Card ? [p0Card] : [],
            isFirst && animate,
        );

        const p2Container = document.getElementById('replay-p2-cards');
        if (state.is_terminal) {
            Cards.renderToContainer(p2Container, p1Card ? [p1Card] : [], false);
        } else {
            Cards.renderToContainer(p2Container, p1Card ? [p1Card] : [], isFirst && animate);
        }

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
                resultEl.textContent = `Player ${state.winner + 1} wins`;
                resultEl.className = 'result-overlay ' + (state.winner === 0 ? 'win' : 'lose');
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
