// API client for Pokerion backend

const API = {
    async train(iterations = 1000, variant = 'kuhn') {
        const res = await fetch(`/api/train?iterations=${iterations}&variant=${variant}`, {
            method: 'POST',
        });
        return res.json();
    },

    async getStrategy(variant = 'kuhn') {
        const res = await fetch(`/api/strategy?variant=${variant}`);
        return res.json();
    },

    async newGame(variant = 'kuhn') {
        const res = await fetch(`/api/game/new?variant=${variant}`, {
            method: 'POST',
        });
        return res.json();
    },

    async takeAction(gameId, action) {
        const res = await fetch(`/api/game/${gameId}/action?action=${action}`, {
            method: 'POST',
        });
        return res.json();
    },

    async newHand(gameId) {
        const res = await fetch(`/api/game/${gameId}/new-hand`, {
            method: 'POST',
        });
        return res.json();
    },

    async getReplay(gameId) {
        const res = await fetch(`/api/replay/${gameId}`);
        return res.json();
    },

    connectTrainingWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        return new WebSocket(`${proto}//${location.host}/api/ws/train`);
    },
};
