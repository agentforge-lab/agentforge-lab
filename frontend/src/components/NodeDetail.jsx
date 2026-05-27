import { useState } from 'react'

const META = {
  planner:          { label: 'Planner',       why: 'Calls the LLM to turn the raw goal into a precise developer brief — exact files, functions, and packages to use.' },
  human_checkpoint: { label: 'Human Review',  why: 'Pauses the pipeline so a human can read the plan before any files are written. Auto-approved in web UI mode.' },
  developer:        { label: 'Developer',     why: 'Calls the LLM to write or patch code files. On retry it receives the full pytest traceback or security report so it can fix the exact failure.' },
  executor:         { label: 'Syntax Check',  why: 'Runs Python\'s compile() on every generated file. A syntax error here sends the developer straight back to the LLM without wasting a test run.' },
  tester:           { label: 'Test Runner',   why: 'Executes pytest against the generated code. Any failure triggers a retry with the full traceback attached so the developer can fix it precisely.' },
  security:         { label: 'Security Scan', why: 'Pattern-matches for hardcoded secrets, dangerous shell patterns, and unvalidated inputs. Blocking findings force a retry rather than committing insecure code.' },
  git_manager:      { label: 'Git Commit',    why: 'Creates a feature branch, stages only the files the agent touched, and commits with a structured message. Never touches .env or key files.' },
}

function duration(start, end) {
  const ms = new Date(end) - new Date(start)
  if (ms < 1000)  return `${ms} ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
}

function StatusBadge({ status }) {
  const colors = {
    idle:      'badge-idle',
    running:   'badge-running',
    completed: 'badge-completed',
    failed:    'badge-failed',
    retrying:  'badge-retrying',
    skipped:   'badge-skipped',
  }
  return <span className={`status-badge ${colors[status] || 'badge-idle'}`}>{status}</span>
}

function LLMCallItem({ call, index }) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`llm-call llm-call-${call.status}`}>
      <button className="llm-call-toggle" onClick={() => setOpen(o => !o)}>
        <span className="llm-model">{call.model}</span>
        <span className="llm-purpose">{(call.purpose || '').slice(0, 42)}</span>
        <span className="llm-meta">
          {call.durationMs > 0 && <span>{(call.durationMs / 1000).toFixed(1)}s</span>}
          {call.costUsd > 0 && <span>${call.costUsd.toFixed(4)}</span>}
          {call.status === 'running' && <span className="pulse-dot" />}
        </span>
        <span className="llm-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {call.tokensIn > 0 && (
        <div className="llm-tokens">
          {call.tokensIn.toLocaleString()} tokens in &nbsp;·&nbsp; {call.tokensOut.toLocaleString()} out
        </div>
      )}

      {open && (
        <div className="llm-call-body">
          {call.promptPreview && (
            <>
              <div className="llm-section-label">Prompt (preview)</div>
              <pre className="llm-preview">{call.promptPreview}</pre>
            </>
          )}
          {call.responsePreview && (
            <>
              <div className="llm-section-label">Response (preview)</div>
              <pre className="llm-preview">{call.responsePreview}</pre>
            </>
          )}
          {call.error && (
            <div className="llm-error">Error: {call.error}</div>
          )}
          {call.status === 'running' && (
            <div className="llm-waiting">Waiting for model response…</div>
          )}
        </div>
      )}
    </div>
  )
}

export function NodeDetail({ nodeId, nodeState }) {
  if (!nodeId || !nodeState) {
    return (
      <div className="detail-empty">
        <div className="detail-empty-arrow">←</div>
        <div>Click any pipeline node to inspect its LLM calls, decisions, and output</div>
      </div>
    )
  }

  const meta = META[nodeId] || { label: nodeId, why: '' }
  const {
    status, llmCalls = [], decisions = [],
    startTime, endTime, data = {}, error, retryCount,
  } = nodeState

  const totalIn    = llmCalls.reduce((s, c) => s + (c.tokensIn  || 0), 0)
  const totalOut   = llmCalls.reduce((s, c) => s + (c.tokensOut || 0), 0)
  const totalCost  = llmCalls.reduce((s, c) => s + (c.costUsd   || 0), 0)
  const totalMs    = llmCalls.reduce((s, c) => s + (c.durationMs|| 0), 0)

  return (
    <div className="node-detail">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="detail-header">
        <div>
          <div className="detail-title">{meta.label}</div>
          <div className="detail-why">{meta.why}</div>
        </div>
        <StatusBadge status={status} />
      </div>

      {/* ── Timing ─────────────────────────────────────────────────── */}
      {startTime && (
        <div className="detail-section">
          <div className="section-title">Timing</div>
          <div className="kv"><span>Started</span><span>{new Date(startTime).toLocaleTimeString()}</span></div>
          {endTime && <div className="kv"><span>Elapsed</span><span>{duration(startTime, endTime)}</span></div>}
        </div>
      )}

      {/* ── Why this node is red / retrying ────────────────────────── */}
      {(error || retryCount > 0) && (
        <div className="detail-section detail-error">
          <div className="section-title">
            {status === 'retrying' ? 'Why this node is retrying' : 'Why this node failed'}
          </div>
          {retryCount > 0 && (
            <div className="retry-count">↺ {retryCount} retr{retryCount === 1 ? 'y' : 'ies'} triggered back to Developer</div>
          )}
          {error && <div className="error-msg">{error}</div>}
          {data.error_preview && (
            <pre className="error-preview">{data.error_preview}</pre>
          )}
        </div>
      )}

      {/* ── Test results ─────────────────────────────────────────────── */}
      {data.testResult && (
        <div className="detail-section">
          <div className="section-title">Test Results</div>
          <div className="kv">
            <span>Passed</span>
            <span style={{ color: '#10B981' }}>{data.testResult.passed} / {data.testResult.total}</span>
          </div>
          {data.testResult.failed > 0 && (
            <div className="kv">
              <span>Failed</span>
              <span style={{ color: '#EF4444' }}>{data.testResult.failed}</span>
            </div>
          )}
          {data.testResult.failures?.map((f, i) => (
            <div key={i} className="test-failure">
              <div className="failure-name">{f.test_name}</div>
              <pre className="failure-error">{(f.error || '').slice(0, 300)}</pre>
            </div>
          ))}
        </div>
      )}

      {/* ── Security findings ────────────────────────────────────────── */}
      {data.findings?.length > 0 && (
        <div className="detail-section">
          <div className="section-title">Security Findings</div>
          {data.findings.map((f, i) => (
            <div key={i} className={`finding finding-${(f.severity || 'info').toLowerCase()}`}>
              <span className="finding-sev">{f.severity}</span>
              <span className="finding-text">{f.issue}</span>
              {f.file && <span className="finding-loc">{f.file}{f.line ? `:${f.line}` : ''}</span>}
            </div>
          ))}
        </div>
      )}

      {/* ── Implementation plan (human checkpoint node) ─────────────── */}
      {nodeId === 'human_checkpoint' && data.developer_brief && (
        <div className="detail-section">
          <div className="section-title">Implementation Plan</div>
          <pre className="plan-brief-detail">{data.developer_brief}</pre>
          {data.branch && (
            <div className="kv" style={{ marginTop: '8px' }}>
              <span>Branch</span><code className="mono">{data.branch}</code>
            </div>
          )}
          {data.working_dir && (
            <div className="kv">
              <span>Workspace</span><code className="mono">{data.working_dir}</code>
            </div>
          )}
          <div className="kv">
            <span>Mode</span>
            <span>{data.mode === 'auto' ? 'Auto-approved (web UI)' : 'Manually approved'}</span>
          </div>
        </div>
      )}

      {/* ── Decisions made ──────────────────────────────────────────── */}
      {decisions.length > 0 && (
        <div className="detail-section">
          <div className="section-title">Decisions Made</div>
          {decisions.map((d, i) => (
            <div key={i} className="decision">
              <div className="decision-label">↳ {d.decision}</div>
              {d.reasoning && <div className="decision-reasoning">{d.reasoning}</div>}
            </div>
          ))}
        </div>
      )}

      {/* ── LLM Calls ────────────────────────────────────────────────── */}
      {llmCalls.length > 0 && (
        <div className="detail-section">
          <div className="section-title">
            LLM Calls
            <span className="section-meta">{llmCalls.length} call{llmCalls.length !== 1 ? 's' : ''}</span>
            {totalCost > 0 && <span className="cost-tag">${totalCost.toFixed(4)}</span>}
          </div>
          {totalIn > 0 && (
            <div className="kv">
              <span>Tokens</span>
              <span>{totalIn.toLocaleString()} in / {totalOut.toLocaleString()} out</span>
            </div>
          )}
          {totalMs > 0 && (
            <div className="kv">
              <span>LLM time</span>
              <span>{(totalMs / 1000).toFixed(1)} s total</span>
            </div>
          )}
          <div className="llm-calls-list">
            {llmCalls.map((call, i) => (
              <LLMCallItem key={call.id || i} call={call} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* ── Output / files ───────────────────────────────────────────── */}
      {data.files?.length > 0 && (
        <div className="detail-section">
          <div className="section-title">Files Changed</div>
          {data.files.map((f, i) => (
            <div key={i} className="file-path">{f}</div>
          ))}
        </div>
      )}
      {data.summary && (
        <div className="detail-section">
          <div className="section-title">Summary</div>
          <div className="output-text">{data.summary}</div>
        </div>
      )}
      {data.commit_sha && (
        <div className="detail-section">
          <div className="kv">
            <span>Commit SHA</span>
            <code className="mono">{data.commit_sha}</code>
          </div>
          {data.branch && (
            <div className="kv">
              <span>Branch</span>
              <code className="mono">{data.branch}</code>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
