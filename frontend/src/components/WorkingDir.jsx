import { useState } from 'react'

async function analyzeDir(path) {
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  return res.json()
}

export function WorkingDir({ onProjectSet, onClear, disabled }) {
  const [picking, setPicking]   = useState(false)   // waiting for OS dialog
  const [analyzing, setAnalyzing] = useState(false) // running explainer agent
  const [analysis, setAnalysis] = useState(null)
  const [error, setError]       = useState(null)
  const [manual, setManual]     = useState(false)
  const [inputPath, setInputPath] = useState('')

  const loading = picking || analyzing

  // ── Open native folder picker ────────────────────────────────────────
  async function handlePick() {
    setPicking(true)
    setError(null)
    try {
      const res = await fetch('/api/pick-folder')
      if (!res.ok) {
        setError(`Server error ${res.status} — restart the backend and try again`)
        return
      }
      const pick = await res.json()
      if (pick.cancelled) return
      if (pick.error)     { setError(pick.error); return }
      setPicking(false)
      await runAnalysis(pick.path)
    } catch {
      setError('Cannot reach AgentForge server — run: agentforge serve')
    } finally {
      setPicking(false)
    }
  }

  // ── Analyze from manual path input ───────────────────────────────────
  async function handleManualAnalyze() {
    const p = inputPath.trim()
    if (!p) return
    setError(null)
    await runAnalysis(p)
  }

  async function runAnalysis(path) {
    setAnalyzing(true)
    setError(null)
    try {
      const data = await analyzeDir(path)
      if (data.error) {
        setError(data.error)
      } else {
        setAnalysis(data)
        setManual(false)
        onProjectSet(data.path)
      }
    } catch {
      setError('Analysis failed — is the server running?')
    } finally {
      setAnalyzing(false)
    }
  }

  function handleClear() {
    setAnalysis(null)
    setError(null)
    setInputPath('')
    setManual(false)
    onClear()
  }

  // ── Analyzed ─────────────────────────────────────────────────────────
  if (analysis) {
    return (
      <div className="wd-bar wd-analyzed">
        <span className="wd-icon">📁</span>
        <span className="wd-path" title={analysis.path}>{analysis.path}</span>
        <span className="wd-sep">·</span>
        <span className="wd-file-count">{analysis.file_count} file{analysis.file_count !== 1 ? 's' : ''}</span>
        {analysis.summary && (
          <>
            <span className="wd-sep">·</span>
            <span className="wd-summary" title={analysis.summary}>
              {analysis.summary.split('\n')[0].slice(0, 120)}
            </span>
          </>
        )}
        <button className="wd-clear" onClick={handleClear} title="Back to sandbox mode">✕</button>
      </div>
    )
  }

  // ── Manual path input ─────────────────────────────────────────────────
  if (manual) {
    return (
      <div className="wd-bar wd-input">
        <span className="wd-icon">📁</span>
        <input
          className="wd-path-input"
          type="text"
          placeholder="/path/to/your/project"
          value={inputPath}
          onChange={e => { setInputPath(e.target.value); setError(null) }}
          onKeyDown={e => e.key === 'Enter' && handleManualAnalyze()}
          autoFocus
          spellCheck={false}
          disabled={loading}
        />
        <button
          className="wd-analyze-btn"
          onClick={handleManualAnalyze}
          disabled={loading || !inputPath.trim()}
        >
          {loading ? '…' : 'Analyze'}
        </button>
        <button className="wd-clear" onClick={() => { setManual(false); setError(null) }}>✕</button>
        {error && <span className="wd-error">{error}</span>}
      </div>
    )
  }

  // ── Default ───────────────────────────────────────────────────────────
  return (
    <div className="wd-bar wd-collapsed">
      <span className="wd-icon">📁</span>
      {analyzing
        ? <span className="wd-sandbox-label">Analyzing project…</span>
        : <span className="wd-sandbox-label">Sandbox mode — new project per run</span>
      }
      {!analyzing && (
        <>
          <button className="wd-open-btn" onClick={handlePick} disabled={disabled || picking}>
            {picking ? 'Waiting for picker…' : 'Browse…'}
          </button>
          <button
            className="wd-open-btn"
            onClick={() => { setManual(true); setError(null) }}
            disabled={disabled || picking}
            style={{ opacity: 0.6 }}
          >
            Type path
          </button>
        </>
      )}
      {error && <span className="wd-error">{error}</span>}
    </div>
  )
}
