import { useEffect, useRef } from 'react'

const TYPE_COLOR = {
  run_started:        '#3B82F6',
  run_completed:      '#10B981',
  run_failed:         '#EF4444',
  run_result:         '#10B981',
  node_entered:       '#60A5FA',
  node_completed:     '#34D399',
  node_failed:        '#F87171',
  node_retrying:      '#FBBF24',
  node_skipped:       '#6B7280',
  llm_call_started:   '#A78BFA',
  llm_call_completed: '#C4B5FD',
  llm_call_failed:    '#F87171',
  decision_made:      '#F472B6',
  test_result:        '#22D3EE',
  security_finding:   '#FB923C',
}

function typeLabel(type) {
  return type.replace(/_/g, ' ')
}

function dataSummary(type, data) {
  if (!data) return ''
  switch (type) {
    case 'llm_call_started':
      return `${data.model} · ${(data.purpose || '').slice(0, 50)}`
    case 'llm_call_completed':
      return `${data.model} · ${data.tokens_in || 0}→${data.tokens_out || 0} tok · ${((data.duration_ms || 0) / 1000).toFixed(1)}s${data.cost_usd > 0 ? ` · $${data.cost_usd.toFixed(4)}` : ''}`
    case 'llm_call_failed':
      return `${data.model} · ${(data.error || '').slice(0, 60)}`
    case 'node_entered':
      return (data.task || '').slice(0, 70)
    case 'node_retrying':
      return `attempt ${data.attempt || '?'} · ${(data.reason || '').slice(0, 50)}`
    case 'node_failed':
      return (data.error || data.reason || data.summary || '').slice(0, 70)
    case 'decision_made':
      return (data.decision || '').slice(0, 80)
    case 'test_result':
      return `${data.passed || 0}/${data.total || 0} passed${data.failed > 0 ? ` — ${data.failed} failed` : ''}`
    case 'security_finding':
      return `[${data.severity || '?'}] ${(data.issue || '').slice(0, 60)}`
    case 'run_result':
      return data.success
        ? `✓ ${data.branch || ''} · ${(data.commit_sha || '').slice(0, 8)}`
        : `✗ ${(data.error || 'failed').slice(0, 60)}`
    default:
      try {
        return JSON.stringify(data).slice(0, 80)
      } catch {
        return ''
      }
  }
}

export function LLMCallLog({ events }) {
  const bodyRef = useRef(null)

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    const el = bodyRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [events.length])

  return (
    <div className="event-log">
      <div className="event-log-header">
        <span>Event Timeline</span>
        <span className="event-count">{events.length} events</span>
      </div>
      <div className="event-log-body" ref={bodyRef}>
        {events.length === 0 ? (
          <div className="event-empty">
            No events yet — enter a goal above and click <strong>Run</strong>.
          </div>
        ) : (
          events.map((evt, i) => (
            <div key={i} className={`event-row event-row-${evt.type}`}>
              <span className="event-seq">#{evt.seq ?? i}</span>
              <span className="event-time">
                {evt.timestamp
                  ? new Date(evt.timestamp).toLocaleTimeString('en', { hour12: false })
                  : '--:--:--'}
              </span>
              <span
                className="event-type"
                style={{ color: TYPE_COLOR[evt.type] || '#94A3B8' }}
              >
                {typeLabel(evt.type)}
              </span>
              {evt.node_id && (
                <span className="event-node">[{evt.node_id.replace(/_/g, ' ')}]</span>
              )}
              <span className="event-data">{dataSummary(evt.type, evt.data)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
