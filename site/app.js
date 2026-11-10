(() => {
  const version = '{{VERSION}}'
  const release = `https://github.com/chrisvoncsefalvay/vespasian/releases/download/v${version}`
  const recordings = {
    'will-climb': {
      label: 'hill-climb',
      caption: 'Follow the objective, inspect generated answers and trace the retained edits.',
      alt: 'Vespasian terminal showing the objective graph, generated answers and model layers'
    },
    setup: {
      label: 'guided setup',
      caption: 'Connect a Hugging Face dataset, inspect the fields and choose the editing controls.',
      alt: 'Vespasian guided setup showing dataset fields and a live record preview'
    }
  }
  let recording = 'will-climb'
  let player = null
  let playerGeneration = 0
  const el = id => document.getElementById(id)

  function tabs(selector, onSelect) {
    const buttons = Array.from(document.querySelectorAll(selector))
    function select(button) {
      buttons.forEach(item => {
        const selected = item === button
        item.setAttribute('aria-selected', String(selected))
        item.tabIndex = selected ? 0 : -1
      })
      onSelect(button)
    }
    buttons.forEach((button, index) => {
      button.addEventListener('click', () => select(button))
      button.addEventListener('keydown', event => {
        const vertical = button.parentElement.getAttribute('aria-orientation') === 'vertical'
        let next
        if (event.key === (vertical ? 'ArrowDown' : 'ArrowRight')) next = (index + 1) % buttons.length
        if (event.key === (vertical ? 'ArrowUp' : 'ArrowLeft')) next = (index - 1 + buttons.length) % buttons.length
        if (event.key === 'Home') next = 0
        if (event.key === 'End') next = buttons.length - 1
        if (next !== undefined) {
          event.preventDefault()
          buttons[next].focus()
          select(buttons[next])
        }
      })
    })
  }

  tabs('[data-recording]', button => {
    recording = button.dataset.recording
    playerGeneration += 1
    if (player) player.dispose()
    player = null
    el('recording-player').replaceChildren()
    el('recording-player').hidden = true
    el('recording-poster').hidden = false
    el('poster-image').src = `media/${recording}-poster.png`
    el('poster-image').alt = recordings[recording].alt
    el('recording-caption').textContent = recordings[recording].caption
    el('recording-download').href = `media/${recording}.cast`
    el('play-recording').setAttribute('aria-label', `Play the ${recordings[recording].label} recording`)
    el('recording-panel').setAttribute('aria-labelledby', button.id)
    el('recording-status').textContent = `${recordings[recording].label} recording selected`
  })

  el('play-recording').addEventListener('click', () => {
    if (!window.AsciinemaPlayer) {
      el('recording-status').textContent = 'Download the recording to play it with asciinema.'
      el('recording-download').focus()
      return
    }
    const generation = ++playerGeneration
    el('recording-poster').hidden = true
    el('recording-player').hidden = false
    player = window.AsciinemaPlayer.create(`media/${recording}.cast`, el('recording-player'), {
      autoPlay: true,
      fit: 'width',
      terminalFontFamily: 'IBM Plex Mono',
      terminalFontSize: '13px',
      terminalLineHeight: 1.2,
      idleTimeLimit: 2,
      controls: true,
      cols: 112,
      rows: 34,
      theme: 'vespasian'
    })
    player.addEventListener('playing', () => {
      if (generation === playerGeneration) el('recording-status').textContent = `${recordings[recording].label} recording playing`
    })
    player.addEventListener('error', () => {
      if (generation !== playerGeneration) return
      el('recording-player').hidden = true
      el('recording-poster').hidden = false
      el('recording-status').textContent = 'Download the recording to play it with asciinema.'
      el('recording-download').focus()
    })
  })

  const datasets = {
    paired: {
      filename: 'examples/concise.jsonl',
      json: "{\"prompt\":\"What is the capital of France? Give only the answer.\",\"target\":\"Paris\",\"split\":\"edit\"}",
      description: 'Choose exact match, token overlap or goal scoring. Score each generated answer against its reference.',
      command: 'vespasian examples/concise.jsonl --goal "Answer precisely" --budget 2'
    },
    corpus: {
      filename: 'examples/style.jsonl',
      json: "{\"text\":\"Rain moved across the courtyard. By the door, a pair of boots waited beneath a coat that had almost dried.\",\"split\":\"edit\"}",
      description: 'Generate unconditionally from a text corpus. Score style and tone against the goal while tracking reference-text loss.',
      command: 'vespasian examples/style.jsonl --dataset-mode corpus --goal "Use spare, measured prose" --budget 2'
    },
    prompts: {
      filename: 'examples/style-prompts.jsonl',
      json: "{\"prompt\":\"Describe a quiet kitchen just before dawn.\",\"split\":\"edit\"}",
      description: 'Use the objective as the scoring rubric. Measure each generated answer against the behaviour you want.',
      command: 'vespasian examples/style-prompts.jsonl --dataset-mode prompts --goal "Write with concrete imagery" --budget 2'
    }
  }
  tabs('[data-mode]', button => {
    const mode = button.dataset.mode
    const data = datasets[mode]
    el('data-example').setAttribute('aria-labelledby', button.id)
    el('data-filename').textContent = data.filename + ' ↗'
    el('data-filename').href = 'https://github.com/chrisvoncsefalvay/vespasian/blob/main/' + data.filename
    el('data-mode-label').textContent = mode.toUpperCase() + ' · SAMPLE'
    el('data-json').textContent = data.json
    el('data-description').textContent = data.description
    el('data-command').textContent = data.command
  })

  const installs = {
    cargo: {
      language: 'SHELL',
      command: 'cargo install vespasian --locked\nvespasian setup\nvespasian',
      note: 'Install Rust and uv, then run these commands in your terminal.',
      link: 'https://crates.io/crates/vespasian',
      label: 'View the Rust crate ↗'
    },
    linux: {
      language: 'SHELL',
      command: `tar -xzf vespasian-${version}-x86_64-unknown-linux-gnu.tar.gz\ncd vespasian-${version}\n./vespasian setup\n./vespasian`,
      note: 'Download the archive and install uv. Linux ARM64 builds are also available in Releases.',
      link: `${release}/vespasian-${version}-x86_64-unknown-linux-gnu.tar.gz`,
      label: 'Download for Linux x64 ↓'
    },
    macos: {
      language: 'SHELL',
      command: `tar -xzf vespasian-${version}-aarch64-apple-darwin.tar.gz\ncd vespasian-${version}\n./vespasian setup\n./vespasian`,
      note: 'Download the Apple silicon archive and install uv, then run these commands.',
      link: `${release}/vespasian-${version}-aarch64-apple-darwin.tar.gz`,
      label: 'Download for macOS Apple silicon ↓'
    },
    windows: {
      language: 'POWERSHELL',
      command: `Expand-Archive vespasian-${version}-x86_64-pc-windows-msvc.zip .\nSet-Location vespasian-${version}\n.\\vespasian.exe setup\n.\\vespasian.exe`,
      note: 'Download the archive and install uv, then open PowerShell. Windows ARM64 builds are also available in Releases.',
      link: `${release}/vespasian-${version}-x86_64-pc-windows-msvc.zip`,
      label: 'Download for Windows x64 ↓'
    }
  }
  tabs('[data-install]', button => {
    const install = installs[button.dataset.install]
    el('install-panel').setAttribute('aria-labelledby', button.id)
    el('install-language').textContent = install.language
    el('install-command').textContent = install.command
    el('install-note').textContent = install.note
    el('platform-download').href = install.link
    el('platform-download').textContent = install.label
    el('copy-install').querySelector('span').textContent = 'Copy'
  })

  el('copy-install').addEventListener('click', async () => {
    const command = el('install-command').textContent
    let copied = false
    try {
      await navigator.clipboard.writeText(command)
      copied = true
    } catch {
      const area = document.createElement('textarea')
      area.value = command
      area.className = 'sr-only'
      document.body.appendChild(area)
      area.select()
      copied = document.execCommand('copy')
      area.remove()
      el('copy-install').focus()
    }
    el('copy-install').querySelector('span').textContent = copied ? 'Copied' : 'Select text'
    el('copy-status').textContent = copied ? 'Installation commands copied' : 'Select and copy the installation commands above'
  })
})()
