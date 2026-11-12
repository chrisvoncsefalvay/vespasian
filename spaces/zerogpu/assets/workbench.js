() => {
  if (window.__vespasianWorkbench) return
  window.__vespasianWorkbench = true
  const guideKey = 'vespasian.guide.v2'
  const themeKey = 'vespasian.theme'
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)')
  let chosenTheme = null
  try { chosenTheme = localStorage.getItem(themeKey) } catch {}
  if (!['light', 'dark'].includes(chosenTheme)) chosenTheme = null
  function applyTheme() {
    const theme = chosenTheme || (systemTheme.matches ? 'dark' : 'light')
    document.documentElement.dataset.vsTheme = theme
    document.querySelectorAll('.gradio-container').forEach(el => { el.dataset.vsTheme = theme })
    document.querySelectorAll('[data-vs-theme-toggle]').forEach(button => {
      const label = theme === 'dark' ? 'Light mode' : 'Dark mode'
      button.textContent = label
      button.setAttribute('aria-label', `Switch to ${label.toLowerCase()}`)
    })
  }
  systemTheme.addEventListener('change', () => { if (!chosenTheme) applyTheme() })
  window.addEventListener('storage', event => {
    if (event.key === themeKey) {
      chosenTheme = ['light', 'dark'].includes(event.newValue) ? event.newValue : null
      applyTheme()
    }
  })
  document.addEventListener('click', event => {
    if (!event.target.closest('[data-vs-theme-toggle]')) return
    chosenTheme = document.documentElement.dataset.vsTheme === 'dark' ? 'light' : 'dark'
    try { localStorage.setItem(themeKey, chosenTheme) } catch {}
    applyTheme()
  })
  applyTheme()
  let dismissed = false, guideStep = 0, returnFocus = null
  let lastTab = null, lastRun = null, restoring = false, focusedControl = null, keyboardTabNavigation = false
  let selection = {turn: null, split: 'all', outcome: 'all', page: 0, follow: true}
  const expanded = new Map(), scrollPositions = new Map()
  const steps = [
    ['Choose a model', 'Select the small model whose weights you want to edit.'],
    ['Set the objective', 'Describe the change and choose its metric and health checks.'],
    ['Connect data', 'Load a Hugging Face dataset or file, map its fields and set example limits.'],
    ['Set the budget', 'Choose the frontier editor and its USD inference budget.'],
    ['Review and run', 'Check the settings, start the run and compare scores and answers.']
  ]
  const visible = el => Boolean(el && el.getClientRects().length && !el.closest('[aria-hidden="true"]'))
  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
  const activeMotion = new Map()
  function reveal(element, name, duration = 200, distance = 4) {
    activeMotion.get(element)?.cancel()
    if (motionPreference.matches || document.hidden || !visible(element) || !element.animate) return
    const animation = element.animate([
      {opacity: .35, transform: `translateY(${distance}px)`},
      {opacity: 1, transform: 'translateY(0)'}
    ], {id: `vs-motion-${name}`, duration, easing: 'cubic-bezier(0.25, 1, 0.5, 1)'})
    activeMotion.set(element, animation)
    const release = () => { if (activeMotion.get(element) === animation) activeMotion.delete(element) }
    animation.finished.then(release, release)
  }
  let lastView = ''
  function syncViewMotion() {
    const monitor = document.querySelector('#vs-monitor')
    const summary = monitor?.querySelector('.vs-result-summary')
    const heading = monitor?.querySelector('.vs-run-heading')
    const view = visible(monitor) ? (summary ? 'complete' : heading ? 'running' : 'pending') : 'setup'
    if (view === lastView) return
    lastView = view
    if (view === 'complete') reveal(summary.querySelector('.vs-section-heading'), 'result', 240)
    else if (view === 'running') reveal(heading, 'monitor')
  }
  let badgeScheduled = false
  function syncPlatformHeader() {
    const badge = document.querySelector('#huggingface-space-header')
    const shown = visible(badge) && getComputedStyle(badge).visibility !== 'hidden'
    const clearance = shown ? Math.min(96, Math.max(0, badge.getBoundingClientRect().bottom) + 8) : 0
    document.documentElement.style.setProperty('--vs-hf-clearance', `${clearance}px`)
  }
  const platformObserver = new MutationObserver(() => {
    if (badgeScheduled) return
    badgeScheduled = true
    requestAnimationFrame(() => { badgeScheduled = false; syncPlatformHeader() })
  })
  const dialog = document.createElement('dialog')
  dialog.id = 'vs-guide'
  dialog.className = 'vs-guide'
  dialog.setAttribute('aria-labelledby', 'vs-guide-title')
  dialog.setAttribute('aria-describedby', 'vs-guide-description')
  function observeVisibleViewport() {
    if (window === window.top || !window.IntersectionObserver) return
    // HF can make its iframe taller than the browser. IntersectionObserver
    // reports clipping across that boundary without reading the parent page.
    const probe = document.createElement('div')
    probe.setAttribute('aria-hidden', 'true')
    probe.style.cssText = 'position:fixed;inset:0;pointer-events:none;opacity:0;z-index:-1'
    document.body.append(probe)
    const intersections = new Map()
    const viewportObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => { if (probe.contains(entry.target)) intersections.set(entry.target, entry.intersectionRect) })
      const shown = [...intersections.values()].filter(rect => rect.width > 0 && rect.height > 0)
      if (!shown.length) return
      const top = Math.max(0, Math.ceil(Math.min(...shown.map(rect => rect.top))))
      const bottom = Math.min(window.innerHeight, Math.floor(Math.max(...shown.map(rect => rect.bottom))))
      const height = Math.max(0, bottom - top)
      const style = document.documentElement.style
      style.setProperty('--vs-visible-top', `${top}px`)
      style.setProperty('--vs-visible-bottom', `${Math.max(0, window.innerHeight - bottom)}px`)
      style.setProperty('--vs-visible-height', `${height}px`)
      dialog.dataset.vsViewport = height < 320 ? 'short' : height < 500 ? 'compact' : 'full'
      if (dialog.open && height < 320 && dialog.querySelector('.vs-guide-top').contains(document.activeElement)) {
        dialog.querySelector('.vs-guide-skip').focus({preventScroll: true})
      }
    }, {threshold: Array.from({length: 129}, (_, index) => index / 128)})
    function resizeProbe() {
      viewportObserver.disconnect()
      intersections.clear()
      probe.replaceChildren()
      // Strips keep reporting parent scroll when the total visible area stays
      // constant, a case a single full-height observation cannot detect.
      const stripHeight = Math.max(64, Math.ceil(window.innerHeight / 64))
      for (let top = 0; top < window.innerHeight; top += stripHeight) {
        const strip = document.createElement('div')
        strip.style.cssText = `position:absolute;left:0;right:0;top:${top}px;height:${Math.min(stripHeight, window.innerHeight - top)}px`
        probe.append(strip)
        viewportObserver.observe(strip)
      }
    }
    resizeProbe()
    window.addEventListener('resize', resizeProbe)
  }
  function syncMotionPreference() {
    dialog.dataset.vsMotion = motionPreference.matches ? 'off' : 'on'
    if (motionPreference.matches) {
      activeMotion.forEach(animation => animation.cancel())
      activeMotion.clear()
    }
  }
  syncMotionPreference()
  motionPreference.addEventListener('change', syncMotionPreference)
  const guideTabs = ['Welcome', 'Method', 'Setup', 'Examples']
  dialog.innerHTML = `<div class="vs-guide-top"><span class="vs-kicker">GETTING STARTED</span><div class="vs-guide-tools"><button type="button" data-vs-theme-toggle aria-label="Switch to light mode">Light mode</button><button type="button" class="vs-guide-close" data-guide-close aria-label="Close guide">×</button></div></div><div class="vs-guide-tabs" role="tablist" aria-label="Getting started sections">${guideTabs.map((label, index) => `<button type="button" role="tab" id="vs-guide-tab-${index}" aria-controls="vs-guide-panel" data-guide-tab="${index}">${label}</button>`).join('')}</div><div class="vs-guide-body" id="vs-guide-panel" role="tabpanel"><div class="vs-guide-index"></div><h2 id="vs-guide-title" tabindex="-1"></h2><p id="vs-guide-description"></p><div class="vs-guide-content"></div><p class="vs-guide-detail"></p><p class="vs-guide-feedback" role="status"></p></div><div class="vs-guide-footer"><button type="button" class="vs-guide-skip" data-guide-close aria-label="Close guide">Close</button><div class="vs-guide-navigation"><button type="button" data-guide-back>Back</button><button type="button" class="vs-guide-next" data-guide-next>Next</button></div></div>`
  const sections = [
    ['Welcome to Vespasian', 'Edit a small model’s weights, measure the change and save the best checkpoint.', 'Sign in to open private datasets and save results to Hugging Face.'],
    ['How editing works', 'The editor inspects the model, proposes a weight change and tests it against your examples.', 'Validation scores decide which edits to keep. Health checks track preserved capabilities. Test data measures the final result.'],
    ['Set up an experiment', 'Choose a model, data, metric and editor budget.', 'Save the retained checkpoint and trajectory locally or to Hugging Face.'],
    ['Choose a benchmark', 'Load a preset, inspect its examples and set your budget.', 'Compare the original and edited model on held-out test questions.']
  ]
  function nativeLogin() {
    const root = document.querySelector('#vs-login')
    return root?.matches('button,a') ? root : root?.querySelector('button,a')
  }
  const isSignedIn = login => Boolean(login && /sign\s*out|log\s*out/i.test(login.textContent))
  function syncGuideLogin() {
    const button = dialog.querySelector('[data-guide-login]')
    if (!button) return
    const login = nativeLogin(), signedIn = isSignedIn(login)
    const label = signedIn ? 'Continue signed in' : 'Sign in with Hugging Face'
    if (button.textContent !== label) button.textContent = label
    const disabled = !login || login.disabled
    if (button.disabled !== Boolean(disabled)) button.disabled = Boolean(disabled)
    const next = dialog.querySelector('.vs-guide-welcome-actions [data-guide-next]')
    const nextLabel = signedIn ? 'Continue to the guide' : 'Proceed without login'
    if (next && next.textContent !== nextLabel) next.textContent = nextLabel
  }
  // Safari blocks the Space's session cookie inside the Hub's cross-site iframe.
  // Open a standalone tab so the callback and app share first-party cookies.
  // The Hub sandbox permits popups but forbids top navigation. Keep Gradio's
  // own OAuth routes and state
  // validation; neither credentials nor session cookies are handled here.
  document.addEventListener('click', event => {
    if (window.self === window.top || !event.target.closest('#vs-login')) return
    const login = nativeLogin()
    // Logout must clear the iframe's own session, including partitioned cookies.
    if (!login || login.disabled || isSignedIn(login)) return
    event.preventDefault()
    event.stopImmediatePropagation()
    const target = new URL('login/huggingface', window.location.origin + '/')
    const returnTo = new URL(window.location.href)
    try {
      if (sessionStorage.getItem(`${guideKey}.resume`) === '1') {
        returnTo.searchParams.set('__vs_guide_resume', '1')
        sessionStorage.removeItem(`${guideKey}.resume`)
      }
    } catch {}
    target.searchParams.set('_target_url', returnTo.pathname + returnTo.search)
    window.open(target.href, '_blank', 'noopener,noreferrer')
    let note = document.querySelector('#vs-login-help')
    if (!note) {
      note = document.createElement('p')
      note.id = 'vs-login-help'
      note.className = 'vs-inline-note'
      note.setAttribute('role', 'status')
      document.querySelector('#vs-account')?.insertAdjacentElement('afterend', note)
    }
    const link = document.createElement('a')
    link.href = window.location.href
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    link.textContent = 'Open standalone workbench'
    note.replaceChildren('Continue in the standalone tab to use your account. ', link)
  }, true)
  function renderGuide(focus = false) {
    dialog.querySelector('.vs-guide-body').scrollTop = 0
    const [title, text, detail] = sections[guideStep]
    dialog.querySelector('.vs-guide-index').textContent = `${guideStep + 1} of ${sections.length}`
    dialog.querySelector('#vs-guide-title').textContent = title
    dialog.querySelector('#vs-guide-description').textContent = text
    dialog.querySelector('.vs-guide-detail').textContent = detail
    dialog.querySelector('.vs-guide-feedback').textContent = ''
    dialog.querySelector('#vs-guide-panel').setAttribute('aria-labelledby', `vs-guide-tab-${guideStep}`)
    dialog.querySelectorAll('[data-guide-tab]').forEach((tab, index) => {
      tab.setAttribute('aria-selected', String(index === guideStep))
      tab.tabIndex = index === guideStep ? 0 : -1
    })
    const content = dialog.querySelector('.vs-guide-content')
    content.replaceChildren()
    if (guideStep === 0) {
      const login = nativeLogin()
      const signedIn = isSignedIn(login)
      content.innerHTML = `<div class="vs-guide-welcome-actions"><button type="button" class="vs-guide-next" data-guide-login>${signedIn ? 'Continue signed in' : 'Sign in with Hugging Face'}</button><button type="button" data-guide-next>${signedIn ? 'Continue to the guide' : 'Proceed without login'}</button></div>`
      syncGuideLogin()
    } else if (guideStep === 1) {
      content.innerHTML = '<ol class="vs-guide-diagram" aria-label="Editing cycle"><li><span>01</span><strong>Assess the baseline</strong></li><li><span>02</span><strong>Inspect the model</strong></li><li><span>03</span><strong>Plan a focused edit</strong></li><li><span>04</span><strong>Patch the weights</strong></li><li><span>05</span><strong>Test the candidate</strong></li><li><span>06</span><strong>Keep or revert</strong></li></ol>'
    } else if (guideStep === 2) {
      const list = document.createElement('ol')
      list.className = 'vs-guide-workflow'
      steps.forEach(([heading, description]) => {
        const item = document.createElement('li'), strong = document.createElement('strong'), text = document.createElement('p')
        strong.textContent = heading
        text.textContent = description
        item.append(strong, text)
        list.append(item)
      })
      content.append(list)
    } else {
      let cards = []
      try { cards = JSON.parse(document.querySelector('#vs-starter-metadata')?.dataset.vsStarters || '[]') } catch {}
      const grid = document.createElement('div')
      grid.className = 'vs-guide-starters'
      cards.forEach(card => {
        const button = document.createElement('button'), title = document.createElement('strong'), description = document.createElement('span'), detail = document.createElement('small')
        button.type = 'button'
        button.dataset.guideStarter = card.id
        title.textContent = card.title
        description.textContent = card.description
        detail.textContent = card.detail || 'Load settings →'
        button.append(title, description, detail)
        grid.append(button)
      })
      if (!cards.length) grid.textContent = 'Starters are unavailable. You can configure an experiment in the workbench.'
      content.append(grid)
    }
    dialog.querySelector('[data-guide-back]').disabled = guideStep === 0
    dialog.querySelector('.vs-guide-navigation [data-guide-next]').textContent = guideStep === sections.length - 1 ? 'Open workbench' : 'Next'
    if (focus) {
      dialog.querySelector('#vs-guide-title').focus({preventScroll: true})
      reveal(dialog.querySelector('.vs-guide-body'), 'guide-step', 180, 3)
    }
  }
  function showGuide(step = 0) {
    if (dialog.open) return
    if (!dialog.isConnected) (document.querySelector('.gradio-container .contain') || document.body).append(dialog)
    activeMotion.get(dialog.querySelector('.vs-guide-body'))?.cancel()
    returnFocus = document.activeElement
    guideStep = step
    renderGuide()
    dialog.showModal()
    applyTheme()
    dialog.querySelector('#vs-guide-title').focus({preventScroll: true})
  }
  function closeGuide() {
    dismissed = true
    try { localStorage.setItem(guideKey, 'dismissed'); sessionStorage.removeItem(`${guideKey}.resume`) } catch {}
    activeMotion.get(dialog.querySelector('.vs-guide-body'))?.cancel()
    dialog.close()
    const heading = [...document.querySelectorAll('#vs-setup .vs-step-intro h2')].find(visible)
    if (visible(returnFocus) && returnFocus !== document.body) returnFocus.focus({preventScroll: true})
    else if (heading) { heading.tabIndex = -1; heading.focus({preventScroll: true}) }
  }
  let loadingStarter = false
  dialog.addEventListener('click', event => {
    const tab = event.target.closest('[data-guide-tab]')
    const starter = event.target.closest('[data-guide-starter]')
    if (tab) { guideStep = Number(tab.dataset.guideTab); renderGuide(); tab.focus() }
    else if (starter && !loadingStarter) {
      const input = document.querySelector('#vs-starter-input textarea, #vs-starter-input input')
      const submit = document.querySelector('#vs-starter-submit')
      if (!input || !submit) { dialog.querySelector('.vs-guide-feedback').textContent = 'The workbench is still loading. Try again in a moment.'; return }
      loadingStarter = true
      starter.disabled = true
      const prototype = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
      Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, starter.dataset.guideStarter)
      input.dispatchEvent(new Event('input', {bubbles: true}))
      requestAnimationFrame(() => { submit.click(); closeGuide(); loadingStarter = false })
    } else if (event.target.closest('[data-guide-login]')) {
      const login = nativeLogin()
      if (!login || login.disabled) return
      if (isSignedIn(login)) { guideStep = 1; renderGuide(true); return }
      // Delegate to Gradio's OAuth component; never read or persist credentials.
      try { sessionStorage.setItem(`${guideKey}.resume`, '1') } catch {}
      dialog.close()
      login.click()
    } else if (event.target.closest('[data-guide-close]')) closeGuide()
    else if (event.target.closest('[data-guide-next]')) {
      if (guideStep === sections.length - 1) closeGuide()
      else { guideStep += 1; renderGuide(true) }
    } else if (event.target.closest('[data-guide-back]') && guideStep > 0) { guideStep -= 1; renderGuide(true) }
    else if (event.target === dialog) {
      const r = dialog.getBoundingClientRect()
      if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) closeGuide()
    }
  })
  dialog.addEventListener('cancel', event => { event.preventDefault(); closeGuide() })
  dialog.addEventListener('keydown', event => {
    const tab = event.target.closest('[data-guide-tab]')
    if (tab && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault()
      guideStep = event.key === 'Home' ? 0 : event.key === 'End' ? 3 : (guideStep + (event.key === 'ArrowRight' ? 1 : -1) + 4) % 4
      renderGuide()
      dialog.querySelector(`[data-guide-tab="${guideStep}"]`).focus()
      return
    }
    if (event.key !== 'Tab') return
    const controls = [...dialog.querySelectorAll('button:not(:disabled),a[href],input,select,textarea')].filter(el => visible(el) && el.tabIndex !== -1)
    const index = controls.indexOf(document.activeElement)
    if ((event.shiftKey && index <= 0) || (!event.shiftKey && (index < 0 || index === controls.length - 1))) {
      event.preventDefault()
      controls[event.shiftKey ? controls.length - 1 : 0]?.focus()
    }
  })
  function resumeGuide() {
    let resume = false
    try { resume = sessionStorage.getItem(`${guideKey}.resume`) === '1'; sessionStorage.removeItem(`${guideKey}.resume`) } catch {}
    const url = new URL(window.location.href)
    if (url.searchParams.get('__vs_guide_resume') === '1') {
      resume = true
      url.searchParams.delete('__vs_guide_resume')
      history.replaceState(history.state, '', url.pathname + url.search + url.hash)
    }
    if (resume) showGuide(1)
    return resume
  }
  window.addEventListener('pageshow', () => { if (document.querySelector('#vs-setup')) resumeGuide() })

  function sendSelection(change) {
    selection = {...selection, ...change}
    if ('turn' in change) { selection.follow = false; selection.page = 0 }
    if ('split' in change || 'outcome' in change) selection.page = 0
    if (change.follow === true) { selection.turn = null; selection.page = 0 }
    const input = document.querySelector('#vs-selection-input textarea, #vs-selection-input input')
    const submit = document.querySelector('#vs-selection-submit')
    if (!input || !submit) return
    const prototype = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, JSON.stringify(selection))
    input.dispatchEvent(new Event('input', {bubbles: true}))
    requestAnimationFrame(() => submit.click())
  }
  function action(control) {
    const name = control.dataset.vsAction, value = control.dataset.vsValue ?? control.value
    if (name === 'follow') sendSelection({follow: value === 'true' || value === '1' || (value == null && !selection.follow)})
    else if (name === 'page') sendSelection({page: Math.max(0, Number(value) || 0)})
    else if (name === 'split' || name === 'outcome') sendSelection({[name]: value})
  }
  function focusTarget(selector) {
    let element
    try { element = document.querySelector(selector) } catch { return }
    if (!visible(element)) {
      if (selector === '#vs-hf-task') {
        const tab = [...document.querySelectorAll('#vs-setup-tabs [role="tab"]')].find(el => el.textContent.includes('02 Behaviour'))
        if (tab && tab.getAttribute('aria-selected') !== 'true') {
          tab.click(); setTimeout(() => focusTarget(selector), 100); return true
        }
      }
      const sections = {
        '#vs-max-turns': '#vs-advanced-runtime', '#vs-reasoning-effort': '#vs-advanced-runtime',
        '#vs-input-price': '#vs-advanced-runtime', '#vs-output-price': '#vs-advanced-runtime',
        '#vs-metric': '#vs-advanced-evaluation', '#vs-target-type': '#vs-advanced-evaluation',
        '#vs-health-checks': '#vs-advanced-evaluation', '#vs-health-count': '#vs-advanced-evaluation',
        '#vs-health-policy': '#vs-advanced-evaluation', '#vs-health-limit': '#vs-advanced-evaluation',
      }
      const section = document.querySelector(sections[selector] || '.__no_advanced_section__')
      const content = section?.querySelector('[data-testid="accordion-content"]')
      if (content && !visible(content)) {
        section.querySelector('button.label-wrap')?.click()
        setTimeout(() => focusTarget(selector), 100); return true
      }
      return false
    }
    if (!element.matches('input,textarea,select,button,a[href],[tabindex]')) {
      element = [...element.querySelectorAll('input,textarea,select,button')].find(visible) || element
    }
    if (!element.matches('input,textarea,select,button,a[href],[tabindex]')) element.tabIndex = -1
    element.scrollIntoView({block: 'center', behavior: 'instant'})
    element.focus({preventScroll: true})
    return true
  }
  document.addEventListener('click', event => {
    if (event.target.closest('[data-vs-help]')) { event.preventDefault(); showGuide(); return }
    const invalid = event.target.closest('[data-vs-focus]')
    if (invalid) { event.preventDefault(); focusTarget(invalid.dataset.vsFocus); return }
    const turn = event.target.closest('[data-vs-turn]')
    if (turn && !turn.disabled) { event.preventDefault(); sendSelection({turn: Number(turn.dataset.vsTurn)}); return }
    const control = event.target.closest('[data-vs-action]')
    if (control && control.tagName !== 'SELECT' && !control.disabled) action(control)
  })
  document.addEventListener('change', event => {
    const control = event.target.closest('select[data-vs-action], input[data-vs-action]')
    if (control) action(control)
  })
  document.addEventListener('keydown', event => {
    const turn = event.target.closest('[data-vs-turn]')
    if (turn && !['BUTTON', 'A'].includes(turn.tagName) && ['Enter', ' '].includes(event.key)) {
      event.preventDefault()
      sendSelection({turn: Number(turn.dataset.vsTurn)})
      return
    }
    const tab = event.target.closest('#vs-setup-tabs [role="tab"]')
    if (!tab || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    const tabs = [...document.querySelectorAll('#vs-setup-tabs [role="tab"]')].filter(visible)
    const index = tabs.indexOf(tab)
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length
    event.preventDefault()
    keyboardTabNavigation = true
    tabs[next].focus()
    tabs[next].click()
  })
  function detailKey(element) {
    return element.dataset.vsDetails || element.id || element.querySelector(':scope > summary')?.textContent.trim().slice(0, 180)
  }
  document.addEventListener('focusin', event => {
    const element = event.target
    if (!element.closest('#vs-dashboard-output')) { focusedControl = null; return }
    if (element.dataset.vsTurn != null) focusedControl = `[data-vs-turn="${CSS.escape(element.dataset.vsTurn)}"]`
    else if (element.dataset.vsAction) focusedControl = `[data-vs-action="${CSS.escape(element.dataset.vsAction)}"]`
  })
  document.addEventListener('toggle', event => {
    const element = event.target
    if (element instanceof HTMLDetailsElement && element.closest('#vs-dashboard-output')) {
      const key = detailKey(element)
      if (key && !restoring) expanded.set(key, element.open)
    }
  }, true)
  document.addEventListener('scroll', event => {
    const element = event.target
    if (!(element instanceof HTMLElement) || !element.closest('#vs-dashboard-output')) return
    const key = element.dataset.vsScroll || (element.matches('.vs-answer-list,.vs-editor-body,.vs-genealogy-scroll,.vs-table-scroll') ? element.className : null)
    if (key && !restoring) scrollPositions.set(key, [element.scrollTop, element.scrollLeft])
  }, true)
  function restoreDashboard() {
    const dashboard = document.querySelector('#vs-dashboard-output .vs-dashboard')
    if (!dashboard) return
    const run = dashboard.dataset.vsRun
    if (run && lastRun && run !== lastRun) { expanded.clear(); scrollPositions.clear() }
    if (run) lastRun = run
    try { selection = {...selection, ...JSON.parse(dashboard.dataset.vsSelection || '{}')} } catch {}
    restoring = true
    dashboard.querySelectorAll('details').forEach(element => {
      const key = detailKey(element)
      if (key && expanded.has(key)) element.open = expanded.get(key)
    })
    dashboard.querySelectorAll('[data-vs-scroll],.vs-answer-list,.vs-editor-body,.vs-genealogy-scroll,.vs-table-scroll').forEach(element => {
      const position = scrollPositions.get(element.dataset.vsScroll || element.className)
      if (position) { element.scrollTop = position[0]; element.scrollLeft = position[1] }
    })
    if (focusedControl && document.activeElement === document.body) dashboard.querySelector(focusedControl)?.focus({preventScroll: true})
    requestAnimationFrame(() => { restoring = false })
  }
  function syncTabs() {
    const container = document.querySelector('#vs-setup-tabs')
    if (!container) return
    const tabs = [...container.querySelectorAll('[role="tab"]')].filter(visible)
    const panels = [...container.querySelectorAll(':scope > .tabitem')]
    const panelFor = (tab, index) => panels.find(panel => panel.id === `vs-step-${tab.dataset.tabId}` || panel.id === tab.getAttribute('aria-controls')) || panels[index]
    tabs.forEach((tab, index) => {
      const panel = panelFor(tab, index)
      if (!panel) return
      if (!tab.id) tab.id = `vs-step-tab-${index}`
      if (!panel.id) panel.id = `vs-step-panel-${index}`
      tab.setAttribute('aria-controls', panel.id)
      panel.setAttribute('role', 'tabpanel')
      panel.setAttribute('aria-labelledby', tab.id)
    })
    const tab = tabs.find(element => element.getAttribute('aria-selected') === 'true')
    if (!tab) return
    const panel = panelFor(tab, tabs.indexOf(tab))
    if (!panel || !visible(panel)) return
    const tabKey = tab.dataset.tabId || tab.id
    if (lastTab && lastTab !== tabKey) {
      if (keyboardTabNavigation) tab.focus({preventScroll: true})
      else {
        const heading = panel.querySelector('h2,h3') || panel
        heading.tabIndex = -1
        heading.scrollIntoView({block: 'start', behavior: 'instant'})
        heading.focus({preventScroll: true})
      }
      reveal(panel.querySelector('.vs-step-intro'), 'setup-step')
      keyboardTabNavigation = false
    }
    lastTab = tabKey
  }
  let scheduled = false, lastError = ''
  const observer = new MutationObserver(() => {
    if (scheduled) return
    scheduled = true
    requestAnimationFrame(() => {
      scheduled = false
      syncGuideLogin()
      syncTabs()
      restoreDashboard()
      syncViewMotion()
      const error = [...document.querySelectorAll('.vs-form-errors[data-vs-focus]')].find(visible)
      const signature = error ? error.textContent + error.dataset.vsFocus : ''
      if (signature && signature !== lastError && focusTarget(error.dataset.vsFocus)) lastError = signature
      else if (!signature) lastError = ''
    })
  })
  function initialise() {
    const container = document.querySelector('.gradio-container')
    if (!container || !document.querySelector('#vs-setup')) { setTimeout(initialise, 100); return }
    // Gradio scopes responsive CSS to .contain; native dialogs still use the top layer.
    (container.querySelector('.contain') || container).append(dialog)
    observeVisibleViewport()
    applyTheme()
    syncPlatformHeader()
    platformObserver.observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['hidden', 'style', 'class']})
    window.addEventListener('resize', syncPlatformHeader)
    syncTabs()
    restoreDashboard()
    syncViewMotion()
    observer.observe(container, {childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['aria-selected', 'hidden', 'style', 'class', 'disabled']})
    try { dismissed = localStorage.getItem(guideKey) === 'dismissed' } catch {}
    if (!resumeGuide() && !dismissed) setTimeout(() => showGuide(), 150)
  }
  initialise()
}
