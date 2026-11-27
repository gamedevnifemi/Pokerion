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

    // The lab only exists where the server has it switched on (local runs).
    // Resolves to null when it does not, so the page can leave the tab hidden.
    async labOptions() {
        const res = await fetch('/api/lab/options');
        return res.ok ? res.json() : null;
    },

    // Plays one match. Resolves to the match, or to { error } with a message.
    async labMatch(request) {
        const res = await fetch('/api/lab/match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });
        const body = await res.json();
        if (res.ok) return body;
        const detail = Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail;
        return { error: body.error || detail || `request failed (${res.status})` };
    },
};
