// The Lab tab: seat bots at a table, play a match on the server, and show it.
//
// One match comes back from the server in one piece (see routes/lab.py):
//   per_hand    what each player expected to win on every hand: the score
//   plays       the hand that was dealt each time
//   references  for an adapting agent: its result without adapting, and the most possible
//   beliefs     what an adapting agent expected of each opponent, through the match
//
// Players keep one number for the whole match. Seats change from hand to hand,
// so "player" and "seat" are never the same thing here.

const Lab = {
    options: null,   // what the server offers: games, kinds of player, limits
    match: null,     // the match on screen
    chart: null,
    hand: 0,         // index of the hand on screen, from 0
    timer: null,     // set while playing through

    // One colour per player, drawn from the site's palette. The first is gold
    // because the first player is the adapting agent by default.
    COLORS: ['#c9a96e', '#7aa7d6', '#d9736b', '#7fb08f', '#b39ddb', '#d6a15f'],
    WINDOW: 24,      // hands in the moving average; a multiple of 2 and of 6, so
                     // it always covers whole cycles of seatings

    // ------------------------------------------------------------ setup

    async init() {
        this.options = await API.labOptions();
        if (!this.options) return;  // the lab is off on this server: stay hidden
        document.getElementById('lab-tab').hidden = false;

        const game = document.getElementById('lab-game');
        for (const g of this.options.games) game.add(new Option(g.name, g.id));
        game.value = this.options.games.at(-1).id;
        game.addEventListener('change', () => this.buildPlayers());
        document.getElementById('lab-hands').max = this.options.max_hands;
        this.buildPlayers();

        document.getElementById('lab-setup').addEventListener('submit', (event) => {
            event.preventDefault();
            this.run();
        });
        document.getElementById('lab-prev').addEventListener('click', () => this.step(-1));
        document.getElementById('lab-next').addEventListener('click', () => this.step(1));
        document.getElementById('lab-play').addEventListener('click', () => this.togglePlay());
        document.getElementById('lab-hand').addEventListener('input', (event) => {
            this.showHand(Number(event.target.value) - 1);
        });
    },

    // One chooser per player at the table, for the game selected.
    buildPlayers() {
        const game = this.options.games.find(g => g.id === document.getElementById('lab-game').value);
        const holder = document.getElementById('lab-players');
        holder.innerHTML = '';

        game.table.forEach((kind, i) => {
            const group = document.createElement('div');
            group.className = 'control-group lab-player';

            const label = document.createElement('label');
            label.htmlFor = `lab-player-${i}`;
            label.textContent = `Player ${i + 1}`;

            const select = document.createElement('select');
            select.id = `lab-player-${i}`;
            for (const k of this.options.kinds) {
                const option = new Option(k.label, k.id);
                option.title = k.description;
                select.add(option);
            }
            select.value = kind;

            // Two settings that only the adapting agent has.
            const extra = document.createElement('div');
            extra.className = 'lab-player-extra';
            extra.innerHTML = `
                <label>Trust in its read
                    <input type="number" class="lab-trust" value="100" min="0" max="100" step="10"
                           title="How often it plays the best reply to what it has seen, instead of solid play."> %
                </label>
                <label>Equal choices go to
                    <select class="lab-ties" title="When two actions look equally good, which one it tries. This decides what it gets to learn.">
                        <option value="bet">betting and calling</option>
                        <option value="check">checking and folding</option>
                    </select>
                </label>`;

            const sync = () => {
                const adapts = this.options.kinds.find(k => k.id === select.value).adapts;
                extra.hidden = !adapts;
            };
            select.addEventListener('change', sync);
            sync();

            group.append(label, select, extra);
            holder.appendChild(group);
        });
    },

    readRequest() {
        const seats = [...document.querySelectorAll('.lab-player')].map(group => ({
            kind: group.querySelector('select').value,
            trust: Number(group.querySelector('.lab-trust').value) / 100,
            ties: group.querySelector('.lab-ties').value,
        }));
        return {
            game: document.getElementById('lab-game').value,
            seats,
            hands: Number(document.getElementById('lab-hands').value),
            reveal: document.getElementById('lab-reveal').value,
            seed: Number(document.getElementById('lab-seed').value),
        };
    },

    async run() {
        const button = document.getElementById('lab-run');
        const status = document.getElementById('lab-status');
        this.stopPlay();
        button.disabled = true;
        status.textContent = 'Playing the match…';

        let match;
        try {
            match = await API.labMatch(this.readRequest());
        } catch (e) {
            match = { error: e.message };
        }
        button.disabled = false;

        if (match.error) {
            status.textContent = `The match was not played: ${match.error}`;
            return;
        }
        status.textContent = '';
        this.match = match;
        document.getElementById('lab-results').hidden = false;
        this.renderLedger();
        this.renderChart();
        this.renderBeliefFrame();
        const slider = document.getElementById('lab-hand');
        slider.max = match.hands;
        this.showHand(0);
    },

    // ------------------------------------------------------------ helpers

    color(player) { return this.COLORS[player % this.COLORS.length]; },

    signed(value, digits = 3) {
        return (value >= 0 ? '+' : '−') + Math.abs(value).toFixed(digits);
    },

    mean(values) { return values.reduce((a, b) => a + b, 0) / values.length; },

    // Each value replaced by the average of itself and the hands before it.
    movingAverage(values) {
        const out = [];
        let sum = 0;
        for (let i = 0; i < values.length; i++) {
            sum += values[i];
            if (i >= this.WINDOW) sum -= values[i - this.WINDOW];
            out.push(sum / Math.min(i + 1, this.WINDOW));
        }
        return out;
    },

    // The adapting agent the reference lines and beliefs are shown for.
    focus() {
        const withReferences = Object.keys(this.match.references);
        if (withReferences.length) return Number(withReferences[0]);
        const adapting = this.match.agents.findIndex(a => a.adapts);
        return adapting >= 0 ? adapting : null;
    },

    dot(player) {
        const dot = document.createElement('span');
        dot.className = 'lab-dot';
        dot.style.background = this.color(player);
        return dot;
    },

    // ------------------------------------------------------------ the score

    renderLedger() {
        const { agents, winnings, per_hand, references } = this.match;
        const holder = document.getElementById('lab-ledger');
        holder.innerHTML = '';
        const recent = Math.min(50, this.match.hands);

        agents.forEach((agent, i) => {
            const row = document.createElement('div');
            row.className = 'lab-ledger-row';

            const name = document.createElement('div');
            name.className = 'lab-ledger-name';
            name.append(this.dot(i), document.createTextNode(agent.name));

            const total = document.createElement('div');
            total.className = 'lab-ledger-value';
            total.textContent = this.signed(winnings[i]);

            row.append(name, total);
            const lines = [[`last ${recent} hands`, this.signed(this.mean(per_hand[i].slice(-recent)))]];

            const reference = references[i];
            if (reference) {
                const gap = reference.ceiling - reference.fixed;
                const late = this.mean(per_hand[i].slice(-recent));
                lines.push(['if it never adapted', this.signed(reference.fixed)]);
                lines.push(['the most possible', this.signed(reference.ceiling)]);
                if (gap > 1e-9) {
                    const share = Math.round(100 * (late - reference.fixed) / gap);
                    lines.push([`share of that gap, last ${recent} hands`, `${share}%`]);
                }
            }
            for (const [label, value] of lines) {
                const detail = document.createElement('div');
                detail.className = 'lab-ledger-detail';
                detail.innerHTML = `<span>${label}</span><span>${value}</span>`;
                row.appendChild(detail);
            }
            holder.appendChild(row);
        });

        const note = document.createElement('p');
        note.className = 'lab-ledger-note';
        note.textContent = 'Chips per hand, over the whole match unless it says otherwise.';
        holder.appendChild(note);
    },

    renderChart() {
        const { agents, per_hand, hands, references } = this.match;
        const labels = Array.from({ length: hands }, (_, i) => i + 1);
        const datasets = agents.map((agent, i) => ({
            label: agent.name,
            data: this.movingAverage(per_hand[i]),
            borderColor: this.color(i),
            backgroundColor: this.color(i),
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
        }));

        const focus = this.focus();
        const reference = focus !== null ? references[focus] : null;
        if (reference) {
            const flat = (label, value, dash) => ({
                label,
                data: labels.map(() => value),
                borderColor: this.color(focus),
                borderWidth: 1,
                borderDash: dash,
                pointRadius: 0,
            });
            datasets.push(flat(`${agents[focus].name}: the most possible`, reference.ceiling, [2, 4]));
            datasets.push(flat(`${agents[focus].name}: if it never adapted`, reference.fixed, [8, 5]));
        }

        const tick = { color: '#57534e', font: { family: "'IBM Plex Mono', monospace", size: 10 } };
        const title = (text) => ({
            display: true, text, color: '#57534e', font: { family: "'Outfit', sans-serif", size: 11 },
        });
        // A thin line at the hand that is on screen below.
        const marker = {
            id: 'handMarker',
            afterDatasetsDraw: (chart) => {
                const x = chart.scales.x.getPixelForValue(this.hand);
                const { top, bottom } = chart.chartArea;
                const ctx = chart.ctx;
                ctx.save();
                ctx.strokeStyle = 'rgba(231, 229, 228, 0.45)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x, top);
                ctx.lineTo(x, bottom);
                ctx.stroke();
                ctx.restore();
            },
        };

        if (this.chart) this.chart.destroy();
        this.chart = new Chart(document.getElementById('lab-chart'), {
            type: 'line',
            data: { labels, datasets },
            plugins: [marker],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                onClick: (event, _elements, chart) => {
                    const hit = chart.getElementsAtEventForMode(event, 'index', { intersect: false }, true);
                    if (hit.length) this.showHand(hit[0].index);
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#a8a29e',
                            font: { family: "'Outfit', sans-serif", size: 11 },
                            boxWidth: 18, boxHeight: 1,
                        },
                    },
                    tooltip: {
                        callbacks: {
                            title: (items) => `Hand ${items[0].label}`,
                            label: (item) => `${item.dataset.label}: ${this.signed(item.parsed.y)}`,
                        },
                    },
                },
                scales: {
                    x: { title: title('Hand'), ticks: { ...tick, maxTicksLimit: 12 },
                         grid: { color: 'rgba(255,255,255,0.03)' } },
                    y: { title: title('Chips per hand'), ticks: tick,
                         grid: { color: 'rgba(255,255,255,0.03)' } },
                },
            },
        });
    },

    // ------------------------------------------------------------ one hand

    step(by) {
        this.stopPlay();
        this.showHand(this.hand + by);
    },

    togglePlay() {
        if (this.timer) return this.stopPlay();
        if (this.hand >= this.match.hands - 1) this.showHand(0);
        document.getElementById('lab-play').textContent = 'Pause';
        this.timer = setInterval(() => {
            if (this.hand >= this.match.hands - 1) return this.stopPlay();
            this.showHand(this.hand + 1);
        }, 450);
    },

    stopPlay() {
        clearInterval(this.timer);
        this.timer = null;
        document.getElementById('lab-play').textContent = 'Play through';
    },

    showHand(index) {
        const { hands, plays, agents } = this.match;
        this.hand = Math.max(0, Math.min(hands - 1, index));
        const play = plays[this.hand];

        document.getElementById('lab-hand').value = this.hand + 1;
        document.getElementById('lab-hand-label').textContent = `Hand ${this.hand + 1} of ${hands}`;
        document.getElementById('lab-prev').disabled = this.hand === 0;
        document.getElementById('lab-next').disabled = this.hand === hands - 1;

        const table = document.getElementById('lab-table');
        table.innerHTML = '';
        play.seating.forEach((player, seat) => {
            const shown = play.shown.includes(seat);
            const box = document.createElement('div');
            box.className = 'lab-seat';
            if (play.winner === seat) box.classList.add('won');

            const where = document.createElement('div');
            where.className = 'lab-seat-where';
            where.textContent = seat === 0 ? `Seat ${seat + 1}, first to act` : `Seat ${seat + 1}`;

            const who = document.createElement('div');
            who.className = 'lab-seat-who';
            who.append(this.dot(player), document.createTextNode(agents[player].name));

            const card = document.createElement('div');
            card.className = 'lab-seat-card';
            card.appendChild(Cards.render(play.cards[seat], true));
            if (!shown) {
                card.classList.add('hidden-card');
                const tag = document.createElement('span');
                tag.className = 'lab-hidden-tag';
                tag.textContent = 'hidden';
                card.appendChild(tag);
            }

            const acts = document.createElement('div');
            acts.className = 'action-timeline lab-seat-actions';
            for (const [actor, action] of play.actions) {
                if (actor !== seat) continue;
                const pill = document.createElement('span');
                pill.className = `action-pill ${action}`;
                pill.textContent = action;
                acts.appendChild(pill);
            }

            const facts = document.createElement('div');
            facts.className = 'lab-seat-facts';
            const chips = play.committed[seat];
            facts.innerHTML =
                `<span>put in ${chips} chip${chips === 1 ? '' : 's'}</span>` +
                `<span>expected ${this.signed(play.expected[seat])}</span>`;

            box.append(where, who, card, acts, facts);
            table.appendChild(box);
        });

        document.getElementById('lab-outcome').textContent = this.describeOutcome(play);
        this.renderBeliefs();
        if (this.chart) this.chart.update('none');
    },

    describeOutcome(play) {
        const name = (seat) => this.match.agents[play.seating[seat]].name;
        const seats = (list) => list.map(s => `seat ${s + 1}`).join(' and ');
        const order = play.actions.map(([seat, action]) => `seat ${seat + 1} ${action}s`).join(', ');
        const won = play.showdown.length
            ? `${name(play.winner)} wins the pot of ${play.pot} at a showdown between ${seats(play.showdown)}.`
            : `${name(play.winner)} wins the pot of ${play.pot} without a showdown.`;
        let seen;
        if (this.match.reveal === 'all') seen = 'Every card was turned over afterwards.';
        else if (play.showdown.length) seen = 'Only their cards were shown.';
        else seen = 'No card was shown.';
        return `In order: ${order}. ${won} ${seen}`;
    },

    // ------------------------------------------------------------ beliefs

    // Splits a key like "K|check:bet" into the card and the betting before it.
    readKey(key) {
        const [card, betting = ''] = key.split('|');
        return { card, betting };
    },

    // The action whose probability a cell shows: the one that puts chips in.
    watchedAction(actions) {
        for (const wanted of ['bet', 'raise', 'call']) {
            const at = actions.indexOf(wanted);
            if (at >= 0) return at;
        }
        return 0;
    },

    // Builds one table per opponent, once per match. renderBeliefs fills it in.
    renderBeliefFrame() {
        const section = document.getElementById('lab-beliefs-section');
        const focus = this.focus();
        const beliefs = focus === null ? null : this.match.beliefs[focus];
        section.hidden = !beliefs;
        if (!beliefs) return;

        const { agents, game } = this.match;
        const infosets = game.infosets.map((infoset, row) => ({ ...infoset, row, ...this.readKey(infoset.key) }));
        const cards = [...new Set(infosets.map(i => i.card))];
        const situations = [...new Set(infosets.map(i => `${i.seat}|${i.betting}`))];

        const holder = document.getElementById('lab-beliefs');
        holder.innerHTML = '';
        for (const other of Object.keys(beliefs.about)) {
            const panel = document.createElement('div');
            panel.className = 'lab-belief-panel';

            const heading = document.createElement('h3');
            heading.append(this.dot(Number(other)), document.createTextNode(`About ${agents[other].name}`));
            panel.appendChild(heading);

            const grid = document.createElement('table');
            grid.className = 'lab-belief-grid';
            const head = grid.createTHead().insertRow();
            head.insertCell().outerHTML = '<th scope="col">When it is</th>';
            for (const card of cards) head.insertCell().outerHTML = `<th scope="col">${card}</th>`;

            const body = grid.createTBody();
            for (const situation of situations) {
                const [seat, betting] = situation.split('|');
                const sample = infosets.find(i => `${i.seat}|${i.betting}` === situation);
                const action = sample.actions[this.watchedAction(sample.actions)];
                const tr = body.insertRow();
                const after = betting ? `after ${betting.split(':').join(', ')}` : 'first to act';
                tr.insertCell().outerHTML =
                    `<th scope="row">seat ${Number(seat) + 1}, ${after}<span>chance it ${action}s</span></th>`;
                for (const card of cards) {
                    const cell = tr.insertCell();
                    const infoset = infosets.find(i => i.card === card && `${i.seat}|${i.betting}` === situation);
                    if (!infoset) continue;
                    cell.className = 'lab-belief-cell';
                    cell.dataset.other = other;
                    cell.dataset.row = infoset.row;
                    cell.dataset.action = this.watchedAction(infoset.actions);
                    cell.innerHTML =
                        '<div class="lab-belief-track"><div class="lab-belief-fill"></div>' +
                        '<div class="lab-belief-truth"></div></div><span class="lab-belief-number"></span>';
                }
            }
            panel.appendChild(grid);
            holder.appendChild(panel);
        }
    },

    // Shows the beliefs as they stood when the hand on screen was about to be played.
    renderBeliefs() {
        const focus = this.focus();
        const beliefs = focus === null ? null : this.match.beliefs[focus];
        if (!beliefs) return;

        // The latest record made at or before this hand.
        let at = 0;
        beliefs.hands_seen.forEach((seen, i) => { if (seen <= this.hand) at = i; });
        const seen = beliefs.hands_seen[at];
        const name = this.match.agents[focus].name;
        const knowsTruth = Object.values(beliefs.about).some(b => b.actual);
        document.getElementById('lab-beliefs-caption').textContent =
            `What ${name} expected of each opponent after its first ${seen} hand${seen === 1 ? '' : 's'}, ` +
            'the latest record before the hand above. ' +
            'Each bar is how likely it thought the opponent was to put chips in, holding that card in that spot.' +
            (knowsTruth ? ' The bright mark is what the opponent really does.' : '');

        for (const cell of document.querySelectorAll('.lab-belief-cell')) {
            const about = beliefs.about[cell.dataset.other];
            const row = Number(cell.dataset.row);
            const action = Number(cell.dataset.action);
            const believed = about.predicted[at][row][action];
            cell.querySelector('.lab-belief-fill').style.width = `${(believed * 100).toFixed(1)}%`;
            cell.querySelector('.lab-belief-number').textContent = `${Math.round(believed * 100)}%`;

            const truth = cell.querySelector('.lab-belief-truth');
            if (about.actual) {
                const actual = about.actual[row][action];
                truth.style.left = `${(actual * 100).toFixed(1)}%`;
                truth.hidden = false;
                cell.title = `believes ${Math.round(believed * 100)}%, really ${Math.round(actual * 100)}%`;
            } else {
                truth.hidden = true;
                cell.title = `believes ${Math.round(believed * 100)}%`;
            }
        }
    },
};
