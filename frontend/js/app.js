// Main app — tab switching

const visited = new Set(['train']); // train is visible on load

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;

            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            const section = document.getElementById(target);
            section.classList.add('active');

            // Only animate on first visit
            if (!visited.has(target)) {
                visited.add(target);
                section.style.animation = 'none';
                void section.offsetWidth;
                section.style.animation = '';
            }
        });
    });

    Training.init();
    Table.init();
    Replay.init();
});
