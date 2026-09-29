(() => {
  const menu = document.querySelector('.menu-toggle');
  const links = document.querySelector('.nav-links');
  menu?.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    links.classList.toggle('is-open', open);
  });
  links?.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
    links.classList.remove('is-open');
    menu.setAttribute('aria-expanded', 'false');
    menu.setAttribute('aria-label', 'Open menu');
  }));

  const tabs = [...document.querySelectorAll('.journey-tab')];
  function activateTab(tab, focus = false) {
    tabs.forEach(item => {
      const active = item === tab;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(item.getAttribute('aria-controls'));
      panel.hidden = !active;
      panel.classList.toggle('is-active', active);
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activateTab(tab));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      activateTab(tabs[next], true);
    });
  });

  document.querySelectorAll('.filter-button').forEach(button => button.addEventListener('click', () => {
    const filter = button.dataset.filter;
    document.querySelectorAll('.filter-button').forEach(item => {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    document.querySelectorAll('.video-card').forEach(card => { card.hidden = filter !== 'all' && card.dataset.category !== filter; });
  }));

  const dialog = document.getElementById('video-dialog');
  const player = document.getElementById('dialog-video');
  const title = document.getElementById('dialog-title');
  const error = document.getElementById('video-error');
  let opener = null;
  document.querySelectorAll('.video-card').forEach(card => card.addEventListener('click', () => {
    opener = card;
    title.textContent = card.dataset.title;
    error.hidden = true;
    player.src = card.dataset.video;
    dialog.showModal();
    player.play().catch(() => {});
  }));
  function closeVideo() { dialog.close(); }
  dialog.querySelector('.dialog-close').addEventListener('click', closeVideo);
  dialog.addEventListener('click', event => { if (event.target === dialog) closeVideo(); });
  dialog.addEventListener('close', () => { player.pause(); player.removeAttribute('src'); player.load(); opener?.focus(); });
  player.addEventListener('error', () => { error.hidden = false; });

  const image = document.querySelector('.phone-frame img');
  image?.addEventListener('error', () => { image.style.display = 'none'; });
  document.getElementById('year').textContent = new Date().getFullYear();
  if ('IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('is-visible'); observer.unobserve(entry.target); } });
    }, { threshold: 0.08, rootMargin: '0px 0px 40px 0px' });
    document.querySelectorAll('.reveal').forEach(item => observer.observe(item));
  } else document.querySelectorAll('.reveal').forEach(item => item.classList.add('is-visible'));
})();
