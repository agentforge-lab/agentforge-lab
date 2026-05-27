import { useState, useRef, useCallback } from 'react'
import { flushSync } from 'react-dom'

// ── Node ordering ─────────────────────────────────────────────────────────
const NODE_IDS = [
  'planner', 'human_checkpoint', 'developer',
  'executor', 'tester', 'security', 'git_manager',
]

function blankNode() {
  return {
    status: 'idle',
    llmCalls: [],
    decisions: [],
    startTime: null,
    endTime: null,
    data: {},
    retryCount: 0,
    error: null,
  }
}

function initialNodes() {
  return Object.fromEntries(NODE_IDS.map(id => [id, blankNode()]))
}

function findActiveNode(nodes) {
  return NODE_IDS.find(id => nodes[id]?.status === 'running') || null
}

// ── Pure state reducer ────────────────────────────────────────────────────
// Lives outside the component so it can be called inside flushSync without
// stale closure issues.  Returns the complete next state.

function applyEvent(prev, raw) {
  const { type, node_id, timestamp, seq, data = {} } = raw
  const events = [...prev.events, raw]

  // Clone nodes shallowly; clone individual nodes only when they change.
  let nodes = prev.nodes

  function cloneNode(id) {
    if (nodes[id]) {
      // Make sure we're working on a fresh copy of the top-level object too
      if (nodes === prev.nodes) nodes = { ...prev.nodes }
      nodes[id] = { ...nodes[id] }
    }
  }

  switch (type) {

    case 'workspace_ready':
      return { ...prev, workspace: data.path, events }

    case 'run_started':
      return {
        ...prev,
        running: true,
        nodes: initialNodes(),
        events: [raw],
        result: null,
        error: null,
      }

    case 'node_entered':
      if (nodes[node_id]) {
        cloneNode(node_id)
        nodes[node_id].status    = 'running'
        nodes[node_id].startTime = timestamp
        nodes[node_id].data      = { ...nodes[node_id].data, ...data }
      }
      break

    case 'node_completed':
      if (nodes[node_id]) {
        cloneNode(node_id)
        nodes[node_id].status  = 'completed'
        nodes[node_id].endTime = timestamp
        nodes[node_id].data    = { ...nodes[node_id].data, ...data }
      }
      break

    case 'node_failed':
      if (nodes[node_id]) {
        cloneNode(node_id)
        nodes[node_id].status  = 'failed'
        nodes[node_id].endTime = timestamp
        nodes[node_id].error   = data.reason || data.error || data.summary || 'Failed'
        nodes[node_id].data    = { ...nodes[node_id].data, ...data }
      }
      break

    case 'node_retrying':
      if (nodes[node_id]) {
        cloneNode(node_id)
        nodes[node_id].status     = 'retrying'
        nodes[node_id].retryCount = Math.max(nodes[node_id].retryCount, (data.attempt || 1) - 1)
        nodes[node_id].error      = data.reason || nodes[node_id].error || ''
        nodes[node_id].data       = { ...nodes[node_id].data, ...data }
      }
      break

    case 'node_skipped':
      if (nodes[node_id]) {
        cloneNode(node_id)
        nodes[node_id].status = 'skipped'
        nodes[node_id].data   = { ...nodes[node_id].data, reason: data.reason }
      }
      break

    // ── LLM calls ────────────────────────────────────────────────────────
    case 'llm_call_started': {
      const tid = node_id || findActiveNode(nodes)
      if (tid && nodes[tid]) {
        cloneNode(tid)
        nodes[tid].llmCalls = [
          ...nodes[tid].llmCalls,
          {
            id:            `${tid}_${seq}`,
            model:         data.model || 'unknown',
            purpose:       data.purpose || '',
            promptPreview: data.prompt_preview || '',
            startTime:     timestamp,
            tokensIn: 0, tokensOut: 0, durationMs: 0, costUsd: 0,
            responsePreview: '',
            status: 'running',
          },
        ]
      }
      break
    }

    case 'llm_call_completed': {
      const tid = node_id || findActiveNode(nodes)
      if (tid && nodes[tid]) {
        cloneNode(tid)
        const calls = [...nodes[tid].llmCalls]
        const idx = calls.reduce(
          (f, c, i) => (c.model === data.model && c.status === 'running' ? i : f), -1,
        )
        if (idx >= 0) {
          calls[idx] = {
            ...calls[idx],
            tokensIn:        data.tokens_in   || 0,
            tokensOut:       data.tokens_out  || 0,
            durationMs:      data.duration_ms || 0,
            costUsd:         data.cost_usd    || 0,
            responsePreview: data.response_preview || '',
            status: 'completed',
          }
        }
        nodes[tid].llmCalls = calls
      }
      break
    }

    case 'llm_call_failed': {
      const tid = node_id || findActiveNode(nodes)
      if (tid && nodes[tid]) {
        cloneNode(tid)
        const calls = [...nodes[tid].llmCalls]
        const idx = calls.reduce(
          (f, c, i) => (c.model === data.model && c.status === 'running' ? i : f), -1,
        )
        if (idx >= 0) calls[idx] = { ...calls[idx], status: 'failed', error: data.error }
        nodes[tid].llmCalls = calls
      }
      break
    }

    case 'decision_made':
      if (nodes[node_id]) {
        cloneNode(node_id)
        nodes[node_id].decisions = [
          ...nodes[node_id].decisions,
          { decision: data.decision, reasoning: data.reasoning },
        ]
      }
      break

    case 'test_result':
      if (nodes[node_id]) {
        cloneNode(node_id)
        nodes[node_id].data = { ...nodes[node_id].data, testResult: data }
      }
      break

    case 'security_finding':
      if (nodes[node_id]) {
        cloneNode(node_id)
        const findings = nodes[node_id].data.findings || []
        nodes[node_id].data = { ...nodes[node_id].data, findings: [...findings, data] }
      }
      break

    // ── Terminal events ───────────────────────────────────────────────────
    case 'run_completed':
      // Pipeline emits this through the event stream — use it as the
      // authoritative result so the banner works even if the server-side
      // run_result send races with the client closing the socket.
      return { ...prev, running: false, result: data, nodes, events }

    case 'run_result':
      // Redundant confirmation from main.py — only apply if we don't have a
      // result yet (run_completed may have arrived first).
      if (!prev.result) {
        return { ...prev, running: false, result: data, nodes, events }
      }
      return { ...prev, events }

    case 'run_failed':
      return { ...prev, running: false, error: data.error || 'Run failed', nodes, events }

    default:
      break
  }

  return { ...prev, nodes, events }
}

// ── Hook ──────────────────────────────────────────────────────────────────

export function useAgentSocket() {
  const [state, setState] = useState({
    connected: false,
    running:   false,
    nodes:     initialNodes(),
    events:    [],
    result:    null,
    error:     null,
    workspace: null,
  })
  const wsRef = useRef(null)

  const connect = useCallback((goal, opts = {}) => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/run`)
    wsRef.current = ws

    ws.onopen = () => {
      // Don't use flushSync here — just a connected flag update
      setState(prev => ({ ...prev, connected: true, error: null }))
      ws.send(JSON.stringify({
        goal,
        auto_approve: opts.autoApprove ?? true,
        max_retries:  opts.maxRetries  ?? 3,
        working_dir:  opts.workingDir  ?? '.',
      }))
    }

    ws.onmessage = (evt) => {
      try {
        const event = JSON.parse(evt.data)
        // Debug: shows every pipeline event in the browser Console tab
        console.log(`[AgentForge] ${event.seq ?? '?'} ${event.type} node=${event.node_id || '-'}`, event.data)
        // flushSync forces React to render immediately for EACH event so the
        // graph animates in real-time instead of jumping to the final state.
        flushSync(() => setState(prev => applyEvent(prev, event)))
      } catch (e) { console.warn('[AgentForge] parse error', e) }
    }

    ws.onclose = () => {
      setState(prev => ({ ...prev, connected: false, running: false }))
      wsRef.current = null
    }

    ws.onerror = () => {
      setState(prev => ({
        ...prev,
        connected: false,
        running: false,
        error: 'Cannot reach AgentForge server — is it running? (agentforge serve)',
      }))
      wsRef.current = null
    }
  }, [])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  return { state, connect, disconnect }
}
