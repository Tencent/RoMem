(() => {
  const demo = document.querySelector('.hero-demo');
  const timeButtons = [...document.querySelectorAll('.time-option')];
  const winner = document.querySelector('.winner-name');
  const obamaValue = document.querySelector('.score-obama-value');
  const trumpValue = document.querySelector('.score-trump-value');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let cycleTimer;

  const modes = {
    past: { winner: 'Obama', obama: '0.92', trump: '0.24' },
    current: { winner: 'Trump', obama: '0.21', trump: '0.94' }
  };

  function setMode(mode, userInitiated = false) {
    if (!demo || !modes[mode]) return;
    demo.dataset.mode = mode;
    timeButtons.forEach((button) => {
      const active = button.dataset.time === mode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    const state = modes[mode];
    winner.innerHTML = `${state.winner} <i>aligned</i>`;
    obamaValue.textContent = state.obama;
    trumpValue.textContent = state.trump;
    if (userInitiated) restartCycle();
  }

  function restartCycle() {
    window.clearInterval(cycleTimer);
    if (!reduceMotion) {
      cycleTimer = window.setInterval(() => {
        setMode(demo.dataset.mode === 'past' ? 'current' : 'past');
      }, 5200);
    }
  }

  timeButtons.forEach((button) => {
    button.addEventListener('click', () => setMode(button.dataset.time, true));
  });
  restartCycle();

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));

  const navToggle = document.querySelector('.nav-toggle');
  const mobileNav = document.querySelector('.mobile-nav');
  function closeMenu() {
    navToggle?.setAttribute('aria-expanded', 'false');
    if (mobileNav) mobileNav.hidden = true;
    document.body.classList.remove('menu-open');
  }
  navToggle?.addEventListener('click', () => {
    const open = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', String(!open));
    mobileNav.hidden = open;
    document.body.classList.toggle('menu-open', !open);
  });
  mobileNav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));

  async function copyTarget(button) {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      const original = button.textContent;
      button.textContent = 'Copied';
      window.setTimeout(() => { button.textContent = original; }, 1600);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  }
  document.querySelectorAll('[data-copy-target]').forEach((button) => {
    button.addEventListener('click', () => copyTarget(button));
  });
})();
