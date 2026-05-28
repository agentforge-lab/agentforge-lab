import { useState } from 'react'
import { DecisionGraph } from './components/DecisionGraph.jsx'
import { NodeDetail } from './components/NodeDetail.jsx'
import { LLMCallLog } from './components/LLMCallLog.jsx'
import { useAgentSocket } from './hooks/useAgentSocket.js'

export default function App() {
  const [goal, setGoal]       = useState('')
  const [planData, setPlanData] = useState(null)   // plan returned by /api/plan
  const [planning, setPlanning] = useState(false)  // waiting for /api/plan response
  const [planError, setPlanError] = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)
  const { state, connect, disconnect } = useAgentSocket()

  const canPlan = goal.trim().length > 3 && !state.running && !planning
  const { running, connected, result, error, nodes, events, workspace } = state
  const totalRetries = Object.values(nodes).reduce((s, n) => s + (n.retryCount || 0), 0)

  // ── Step 1: fetch plan from /api/plan ────────────────────────────────────
  async function handlePlan() {
    if (!canPlan) return
    setPlanData(null)
    setPlanError(null)
    setPlanning(true)
    setSelectedNode(null)
    try {
      const res = await fetch('/api/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal: goal.trim() }),
      })
      const data = await res.json()
      if (data.error) {
        setPlanError(data.error)
      } else {
        setPlanData(data)
      }
    } catch (e) {
      setPlanError('Failed to reach AgentForge server — is it running?')
    } finally {
      setPlanning(false)
    }
  }

  // ── Step 2: approve plan → start WebSocket pipeline ─────────────────────
  function handleApprove() {
    if (!planData) return
    setPlanData(null)
    connect(goal.trim(), { autoApprove: true, maxRetries: 3 })
  }

  function handleCancel() {
    setPlanData(null)
    setPlanError(null)
  }

  function handleStop() {
    disconnect()
    setPlanData(null)
  }

  return (
    <div className="app">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="brand">
          <span className="brand-icon">⚙</span>
          <span className="brand-name">AgentForge</span>
          <span className="brand-sub">Decision Graph</span>
        </div>

        <div className="goal-area">
          <input
            className="goal-input"
            type="text"
            placeholder='e.g. "build a CLI calculator in Python"'
            value={goal}
            onChange={e => setGoal(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handlePlan()}
            disabled={running || planning}
            spellCheck={false}
          />
          {running
            ? <button className="btn-stop" onClick={handleStop}>■ Stop</button>
            : <button className="btn-run" onClick={handlePlan} disabled={!canPlan}>
                {planning ? '…' : '▶ Run'}
              </button>
          }
        </div>

        <div className="header-status">
          <span className={`ws-dot ${running ? 'ws-running' : connected ? 'ws-connected' : ''}`} />
          <span className="ws-label">
            {planning ? 'Planning…' : running ? 'Running' : connected ? 'Connected' : 'Ready'}
          </span>
          {totalRetries > 0 && (
            <span className="retry-badge" title="LLM retries triggered">↺ {totalRetries}</span>
          )}
        </div>
      </header>

      {/* ── Planning in progress ────────────────────────────────────── */}
      {planning && (
        <div className="plan-review plan-loading">
          <div className="plan-review-header">
            <span className="plan-review-title">⏳ Generating implementation plan…</span>
          </div>
          <div className="plan-loading-steps">
            <div className="plan-loading-step plan-step-active">⚙ Planner agent started</div>
            <div className="plan-loading-step plan-step-active">🧠 Calling qwen2.5-coder:7b…</div>
            <div className="plan-loading-step plan-step-pending">📋 Structuring implementation spec</div>
            <div className="plan-loading-step plan-step-pending">⎇ Preparing feature branch</div>
          </div>
          <div className="plan-loading-note">This takes 5–15 seconds. You'll review the plan before anything runs.</div>
        </div>
      )}

      {/* ── Plan review panel ───────────────────────────────────────── */}
      {planData && (
        <div className="plan-review">
          <div className="plan-review-header">
            <span className="plan-review-title">Implementation Plan</span>
            <span className="plan-review-branch">⎇ {planData.branch}</span>
          </div>
          <div className="plan-brief">{planData.developer_brief || '(No brief generated — will use raw goal)'}</div>
          <div className="plan-meta">
            <span>📁 {planData.workspace_parent}/…</span>
            {planData.nodes && (
              <span>{Object.keys(planData.nodes).length} pipeline nodes</span>
            )}
          </div>
          <div className="plan-actions">
            <button className="btn-approve" onClick={handleApprove}>✓ Approve &amp; Run</button>
            <button className="btn-cancel"  onClick={handleCancel}>✕ Cancel</button>
          </div>
        </div>
      )}

      {/* ── Banners ─────────────────────────────────────────────────── */}
      {planError && <div className="banner banner-error">⚠ {planError}</div>}
      {workspace && !result && (
        <div className="banner banner-workspace">
          📁 Writing to: <code>{workspace}</code>
        </div>
      )}
      {result && (
        <div className={`banner ${result.success ? 'banner-success' : 'banner-failure'}`}>
          {result.success ? (
            <>
              ✓ Complete
              {result.branch      && <> · Branch: <code>{result.branch}</code></>}
              {result.commit_sha  && <> · Commit: <code>{result.commit_sha.slice(0, 8)}</code></>}
              {result.tests_passed    && <> · Tests passed</>}
              {result.security_passed && <> · Security passed</>}
              {result.retry_count > 0 && <> · {result.retry_count} retr{result.retry_count === 1 ? 'y' : 'ies'}</>}
            </>
          ) : (
            <>✗ Failed{result.error ? ` — ${result.error}` : ''}</>
          )}
        </div>
      )}
      {error && <div className="banner banner-error">⚠ {error}</div>}

      {/* ── Main layout: graph + detail ─────────────────────────────── */}
      <main className="app-main">
        <section className="panel panel-graph">
          <div className="panel-header">Pipeline</div>
          <DecisionGraph
            nodes={nodes}
            selectedNode={selectedNode}
            onSelectNode={setSelectedNode}
          />
        </section>

        <section className="panel panel-detail">
          <div className="panel-header">
            {selectedNode
              ? selectedNode.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
              : 'Node Detail'}
          </div>
          <div className="panel-body">
            <NodeDetail
              nodeId={selectedNode}
              nodeState={selectedNode ? nodes[selectedNode] : null}
            />
          </div>
        </section>
      </main>

      {/* ── Event timeline ──────────────────────────────────────────── */}
      <LLMCallLog events={events} />
    </div>
  )
}
