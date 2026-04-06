// Strategy visualization using CSS bars (smoother than Chart.js)

const StrategyCharts = {
    charts: {}, // keyed by info set key -> DOM element

    renderAll(strategy, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        this.charts = {};

        const keys = Object.keys(strategy).sort();

        keys.forEach((key, i) => {
            const actions = strategy[key];
            const card = document.createElement('div');
            card.className = 'chart-card';
            card.style.animationDelay = `${i * 0.03}s`;
            card.style.animation = 'fadeIn 0.4s var(--ease-out-expo) both';

            const title = document.createElement('h3');
            title.textContent = key;
            card.appendChild(title);

            const bars = document.createElement('div');
            bars.className = 'strat-bars';

            for (const [action, prob] of Object.entries(actions)) {
                const row = document.createElement('div');
                row.className = 'strat-row';

                const label = document.createElement('span');
                label.className = 'strat-action';
                label.textContent = action;

                const track = document.createElement('div');
                track.className = 'strat-track';

                const fill = document.createElement('div');
                fill.className = `strat-fill ${action}`;
                fill.style.width = `${(prob * 100).toFixed(1)}%`;
                fill.dataset.action = action;

                const pct = document.createElement('span');
                pct.className = 'strat-pct';
                pct.textContent = `${(prob * 100).toFixed(0)}%`;
                pct.dataset.action = action;

                track.appendChild(fill);
                row.appendChild(label);
                row.appendChild(track);
                row.appendChild(pct);
                bars.appendChild(row);
            }

            card.appendChild(bars);
            container.appendChild(card);
            this.charts[key] = card;
        });
    },

    updateAll(strategy) {
        for (const [key, actions] of Object.entries(strategy)) {
            const card = this.charts[key];
            if (!card) continue;

            for (const [action, prob] of Object.entries(actions)) {
                const fill = card.querySelector(`.strat-fill[data-action="${action}"]`);
                const pct = card.querySelector(`.strat-pct[data-action="${action}"]`);
                if (fill) fill.style.width = `${(prob * 100).toFixed(1)}%`;
                if (pct) pct.textContent = `${(prob * 100).toFixed(0)}%`;
            }
        }
    },
};
