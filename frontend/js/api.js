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

    async newMatch(variant = 'kuhn', length = 50) {
        const res = await fetch(`/api/match/new?variant=${variant}&length=${length}`, {
            method: 'POST',
        });
        return res.json();
    },

    async takeAction(matchId, action) {
        const res = await fetch(`/api/match/${matchId}/action?action=${action}`, {
            method: 'POST',
        });
        return res.json();
    },

    async newHand(matchId) {
        const res = await fetch(`/api/match/${matchId}/new-hand`, {
            method: 'POST',
        });
        return res.json();
    },

    async getMatchState(matchId) {
        const res = await fetch(`/api/match/${matchId}/state`);
        return res.json();
    },

    async getReplay(matchId) {
        const res = await fetch(`/api/replay/${matchId}`);
        return res.json();
    },

    async listSessions() {
        const res = await fetch('/api/replay/sessions');
        return res.json();
    },

    connectTrainingWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        return new WebSocket(`${proto}//${location.host}/api/ws/train`);
    },
};
