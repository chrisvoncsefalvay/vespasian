() => {
  if (window.__vespasianWorkbench) return
  window.__vespasianWorkbench = true
  const guideKey = 'vespasian.guide.v1'
  let dismissed = false, guideStep = 0, returnFocus = null
  let lastTab = null, lastRun = null, restoring = false, focusedControl = null, keyboardTabNavigation = false
  let selection = {turn: null, split: 'all', outcome: 'all', page: 0, follow: true}
  const expanded = new Map(), scrollPositions = new Map()
  const steps = [
    ['Choose the small model to edit', 'Select the target model first, then define its desired behaviour and connect examples. The frontier editor is selected separately in Runtime.', ['Target model', 'Desired behaviour', 'Examples'], 'Target model availability depends on this Space and your Hugging Face access.'],
    ['Define the behaviour you want to change', 'Describe a successful result, choose how to measure it and select any capabilities you want to preserve. Vespasian uses these decisions to compare proposed weight changes.', ['Your objective', 'Success metric', 'Preservation checks'], 'You can edit every setting before the experiment starts.'],
    ['Connect examples that support the objective', 'Use paired questions and answers, reference writing for style, or prompts with a scoring rubric. Connect a Hugging Face dataset or upload a file, then inspect the mapped examples and choose how many to use.', ['Choose data', 'Inspect examples', 'Set the limit'], 'Signing in gives you access to your private datasets and Hub destinations.'],
    ['Choose an editor and a spending limit', 'The editor model proposes changes to the small model’s tensors. Your USD budget covers the editor’s API calls and starts at $10. Reasoning defaults to low; a turn cap is optional. Hugging Face supplies the GPU time used to evaluate those changes.', ['Editor model', 'USD budget', 'Review and start'], 'The final review brings your data, scoring rules and budget together.'],
    ['Follow each edit back to its evidence', 'Select a turn to inspect its score, changed layers and acceptance decision. Compare the original and edited answers, follow the retained checkpoint, then download the result or save it to your Hub account.', ['Select a turn', 'Compare outputs', 'Save the result'], 'Reopen this guide at any time from Help.']
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
  function syncMotionPreference() {
    dialog.dataset.vsMotion = motionPreference.matches ? 'off' : 'on'
    if (motionPreference.matches) {
      activeMotion.forEach(animation => animation.cancel())
      activeMotion.clear()
    }
  }
  syncMotionPreference()
  motionPreference.addEventListener('change', syncMotionPreference)
  dialog.innerHTML = `<div class="vs-guide-top"><span class="vs-kicker">GETTING STARTED</span><button type="button" class="vs-guide-close" data-guide-close aria-label="Close guide">×</button></div><div class="vs-guide-body"><div class="vs-guide-index" aria-live="polite"></div><h2 id="vs-guide-title" tabindex="-1"></h2><p id="vs-guide-description"></p><ol class="vs-guide-diagram" aria-label="Workflow"></ol><p class="vs-guide-detail"></p></div><div class="vs-guide-footer"><button type="button" class="vs-guide-skip" data-guide-close>Skip guide</button><div class="vs-guide-navigation"><button type="button" data-guide-back>Back</button><button type="button" class="vs-guide-next" data-guide-next>Next</button></div></div>`
  function renderGuide(focus = false) {
    const [title, text, labels, detail] = steps[guideStep]
    dialog.querySelector('.vs-guide-index').textContent = `${guideStep + 1} of ${steps.length}`
    dialog.querySelector('#vs-guide-title').textContent = title
    dialog.querySelector('#vs-guide-description').textContent = text
    dialog.querySelector('.vs-guide-detail').textContent = detail
    dialog.querySelector('.vs-guide-diagram').replaceChildren(...labels.map((label, index) => {
      const item = document.createElement('li'), number = document.createElement('span'), text = document.createElement('strong')
      number.textContent = String(index + 1).padStart(2, '0')
      text.textContent = label
      item.append(number, text)
      return item
    }))
    dialog.querySelector('[data-guide-back]').disabled = guideStep === 0
    dialog.querySelector('[data-guide-next]').textContent = guideStep === steps.length - 1 ? 'Configure experiment' : 'Next'
    if (focus) {
      dialog.querySelector('#vs-guide-title').focus({preventScroll: true})
      reveal(dialog.querySelector('.vs-guide-body'), 'guide-step', 180, 3)
    }
  }
  function showGuide() {
    if (!dialog.isConnected) document.body.append(dialog)
    activeMotion.get(dialog.querySelector('.vs-guide-body'))?.cancel()
    returnFocus = document.activeElement
    guideStep = 0
    renderGuide()
    if (!dialog.open) dialog.showModal()
    dialog.querySelector('#vs-guide-title').focus({preventScroll: true})
  }
  function closeGuide() {
    dismissed = true
    try { localStorage.setItem(guideKey, 'dismissed') } catch {}
    activeMotion.get(dialog.querySelector('.vs-guide-body'))?.cancel()
    dialog.close()
    const heading = [...document.querySelectorAll('#vs-setup .vs-step-intro h2')].find(visible)
    if (visible(returnFocus) && returnFocus !== document.body) returnFocus.focus({preventScroll: true})
    else if (heading) { heading.tabIndex = -1; heading.focus({preventScroll: true}) }
  }
  dialog.addEventListener('click', event => {
    if (event.target.closest('[data-guide-close]')) closeGuide()
    else if (event.target.closest('[data-guide-next]')) {
      if (guideStep === steps.length - 1) closeGuide()
      else { guideStep += 1; renderGuide(true) }
    } else if (event.target.closest('[data-guide-back]') && guideStep > 0) { guideStep -= 1; renderGuide(true) }
    else if (event.target === dialog) {
      const r = dialog.getBoundingClientRect()
      if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) closeGuide()
    }
  })
  dialog.addEventListener('cancel', event => { event.preventDefault(); closeGuide() })
  dialog.addEventListener('keydown', event => {
    if (event.key !== 'Tab') return
    const controls = [...dialog.querySelectorAll('button:not(:disabled),a[href],input,select,textarea')].filter(visible)
    const index = controls.indexOf(document.activeElement)
    if ((event.shiftKey && index <= 0) || (!event.shiftKey && (index < 0 || index === controls.length - 1))) {
      event.preventDefault()
      controls[event.shiftKey ? controls.length - 1 : 0]?.focus()
    }
  })

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
    document.body.append(dialog)
    syncPlatformHeader()
    platformObserver.observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['hidden', 'style', 'class']})
    window.addEventListener('resize', syncPlatformHeader)
    syncTabs()
    restoreDashboard()
    syncViewMotion()
    observer.observe(container, {childList: true, subtree: true, attributes: true, attributeFilter: ['aria-selected', 'hidden', 'style', 'class']})
    try { dismissed = localStorage.getItem(guideKey) === 'dismissed' } catch {}
    if (!dismissed) setTimeout(showGuide, 150)
  }
  initialise()
}
