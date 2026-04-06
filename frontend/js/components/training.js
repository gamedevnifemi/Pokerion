// Training dashboard logic

const Training = {
    ws: null,
    convergenceChart: null,
    convergenceData: [],

    init() {
        document.getElementById('train-btn').addEventListener('click', () => this.startTraining());
        this._initConvergenceChart();
    },

    async startTraining() {
        const btn = document.getElementById('train-btn');
        const status = document.getElementById('train-status');
        const iterations = parseInt(document.getElementById('train-iterations').value);
        const variant = document.getElementById('train-variant').value;

        btn.disabled = true;
        status.textContent = 'connecting...';

        this.ws = API.connectTrainingWS();

        this.ws.onopen = () => {
            status.textContent = 'training...';
            this.ws.send(JSON.stringify({
                iterations,
                variant,
                batch_size: Math.max(10, Math.floor(iterations / 50)),
            }));
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'progress') {
                this._updateMetrics(data);
                this._updateConvergence(data);

                if (Object.keys(StrategyCharts.charts).length === 0) {
                    StrategyCharts.renderAll(data.strategy, 'strategy-charts');
                } else {
                    StrategyCharts.updateAll(data.strategy);
                }
            }

            if (data.type === 'done') {
                status.textContent = `${data.iteration} iterations`;
                btn.disabled = false;
                this.ws.close();
            }
        };

        this.ws.onerror = () => {
            status.textContent = 'WS error, using REST...';
            this.ws.close();
            this._trainREST(iterations, variant, btn, status);
        };

        this.ws.onclose = () => {
            if (btn.disabled) btn.disabled = false;
        };
    },

    async _trainREST(iterations, variant, btn, status) {
        try {
            const result = await API.train(iterations, variant);
            this._updateMetrics({
                iteration: result.total_iterations,
                exploitability: result.snapshots.at(-1)?.exploitability ?? 0,
                game_values: result.snapshots.at(-1)?.game_values ?? [0, 0],
            });
            StrategyCharts.renderAll(result.strategy, 'strategy-charts');

            for (const snap of result.snapshots) {
                this._updateConvergence(snap);
            }

            status.textContent = `${result.total_iterations} iterations`;
        } catch (e) {
            status.textContent = `Error: ${e.message}`;
        }
        btn.disabled = false;
    },

    _updateMetrics(data) {
        this._animateValue('metric-iterations', data.iteration);
        document.getElementById('metric-exploitability').textContent =
            data.exploitability.toFixed(6);
        document.getElementById('metric-game-value').textContent =
            data.game_values[0].toFixed(4);
    },

    _animateValue(id, target) {
        const el = document.getElementById(id);
        el.textContent = target.toLocaleString();
    },

    _initConvergenceChart() {
        const ctx = document.getElementById('convergence-chart');
        this.convergenceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Exploitability',
                    data: [],
                    borderColor: '#c9a96e',
                    backgroundColor: 'rgba(201, 169, 110, 0.08)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#a8a29e',
                            font: { family: "'Outfit', sans-serif", size: 11 },
                        },
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true, text: 'Iteration',
                            color: '#57534e',
                            font: { family: "'Outfit', sans-serif", size: 11 },
                        },
                        ticks: {
                            color: '#57534e',
                            font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                    },
                    y: {
                        title: {
                            display: true, text: 'Exploitability',
                            color: '#57534e',
                            font: { family: "'Outfit', sans-serif", size: 11 },
                        },
                        ticks: {
                            color: '#57534e',
                            font: { family: "'IBM Plex Mono', monospace", size: 10 },
                        },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        min: 0,
                    },
                },
            },
        });
    },

    _updateConvergence(data) {
        if (!data.iteration || data.exploitability === undefined) return;
        this.convergenceData.push({
            iteration: data.iteration,
            exploitability: data.exploitability,
        });

        this.convergenceChart.data.labels = this.convergenceData.map(d => d.iteration);
        this.convergenceChart.data.datasets[0].data = this.convergenceData.map(d => d.exploitability);
        this.convergenceChart.update('none');
    },
};
