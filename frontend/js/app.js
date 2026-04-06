// Main app — tab switching with transitions

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;

            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(c => {
                c.classList.remove('active');
                c.style.animation = 'none';
            });

            const section = document.getElementById(target);
            section.classList.add('active');
            // Re-trigger animation
            void section.offsetWidth;
            section.style.animation = '';
        });
    });

    // Initialize components
    Training.init();
    Table.init();
    Replay.init();
});
