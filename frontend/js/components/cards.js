// Card rendering with 3D flip animations

const SUIT_MAP = {
    J: { suit: '\u2660', name: 'spade' },   // Jack of Spades
    Q: { suit: '\u2665', name: 'heart' },    // Queen of Hearts
    K: { suit: '\u2666', name: 'diamond' },  // King of Diamonds
};

const Cards = {
    render(card, faceUp = true, animate = false) {
        const wrapper = document.createElement('div');
        wrapper.className = 'card-wrapper';
        if (animate) wrapper.classList.add('dealing');

        const inner = document.createElement('div');
        inner.className = 'card-inner';
        if (faceUp && card && card !== '?') inner.classList.add('revealed');

        // Back
        const back = document.createElement('div');
        back.className = 'card-back';

        // Front
        const front = document.createElement('div');
        front.className = 'card-face';

        if (card && card !== '?') {
            const info = SUIT_MAP[card] || { suit: '?', name: '' };
            front.classList.add(card);

            front.innerHTML = `
                <span class="card-rank-top">${card}</span>
                <span class="card-suit-top">${info.suit}</span>
                <span class="card-center-suit">${info.suit}</span>
                <span class="card-suit-bottom">${info.suit}</span>
                <span class="card-rank-bottom">${card}</span>
            `;
        }

        inner.appendChild(back);
        inner.appendChild(front);
        wrapper.appendChild(inner);
        return wrapper;
    },

    renderToContainer(container, cards, animate = false) {
        container.innerHTML = '';
        cards.forEach((card, i) => {
            const faceUp = card && card !== '?';
            const el = this.render(card, faceUp, animate);
            if (animate) {
                el.style.animationDelay = `${i * 0.15}s`;
            }
            container.appendChild(el);
        });
    },

    revealAll(container) {
        container.querySelectorAll('.card-inner').forEach((inner, i) => {
            setTimeout(() => inner.classList.add('revealed'), i * 200);
        });
    },
};
