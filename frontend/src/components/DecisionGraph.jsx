// Pipeline decision graph — fixed SVG layout, no force simulation needed.

const NODE_DEFS = [
  { id: 'planner',          label: 'Planner',       icon: '🧠', desc: 'LLM creates implementation plan'          },
  { id: 'human_checkpoint', label: 'Human Review',  icon: '👤', desc: 'Approve before writing any code'          },
  { id: 'developer',        label: 'Developer',     icon: '⚙️',  desc: 'LLM writes / patches code files'         },
  { id: 'executor',         label: 'Syntax Check',  icon: '🔍', desc: 'Validates Python syntax & imports'        },
  { id: 'tester',           label: 'Test Runner',   icon: '🧪', desc: 'Runs pytest — failure triggers retry'     },
  { id: 'security',         label: 'Security Scan', icon: '🛡️', desc: 'Detects secrets & dangerous patterns'     },
  { id: 'git_manager',      label: 'Git Commit',    icon: '📦', desc: 'Branches, stages, and commits result'     },
]

const STATUS_STYLE = {
  idle:      { fill: '#080F1C', stroke: '#1E3A58', labelColor: '#4A6480', subColor: '#2A3F55' },
  running:   { fill: '#061640', stroke: '#3B82F6', labelColor: '#93C5FD', subColor: '#60A5FA' },
  completed: { fill: '#041D10', stroke: '#16A34A', labelColor: '#4ADE80', subColor: '#22C55E' },
  failed:    { fill: '#1A0505', stroke: '#DC2626', labelColor: '#FCA5A5', subColor: '#F87171' },
  retrying:  { fill: '#1A0B00', stroke: '#D97706', labelColor: '#FDE68A', subColor: '#FCD34D' },
  skipped:   { fill: '#080F1C', stroke: '#2A3F55', labelColor: '#4A6480', subColor: '#2A3F55' },
}

// ── Layout constants ───────────────────────────────────────────────────────
const SVG_W   = 500
const NODE_W  = 390
const NODE_H  = 68
const CX      = SVG_W / 2                   // 250
const NX      = CX - NODE_W / 2             // 55  (left edge of nodes)
const NR      = CX + NODE_W / 2             // 445 (right edge of nodes)
const SPACING = 124
const Y0      = 88

const nodeCount = NODE_DEFS.length
const SVG_H = Y0 + (nodeCount - 1) * SPACING + NODE_H / 2 + 64

const cy = (i) => Y0 + i * SPACING

// ── Helpers ────────────────────────────────────────────────────────────────
function duration(start, end) {
  const ms = new Date(end) - new Date(start)
  if (ms < 1000)  return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.floor(ms / 60000)}m${Math.floor((ms % 60000) / 1000)}s`
}

function subLabel(ns) {
  switch (ns.status) {
    case 'running':   return 'Running…'
    case 'completed': return ns.data?.commit_sha
      ? `Commit ${ns.data.commit_sha.slice(0, 8)}`
      : ns.data?.summary
        ? ns.data.summary.slice(0, 46)
        : 'Done'
    case 'failed':    return (ns.error || 'Failed').slice(0, 46)
    case 'retrying':  return `Retry ${ns.retryCount + 1} — ${(ns.error || '').slice(0, 36)}`
    case 'skipped':   return `Skipped${ns.data?.reason ? ': ' + ns.data.reason.slice(0, 36) : ''}`
    default:          return NODE_DEFS.find(n => n.id === ns.id)?.desc || 'Waiting'
  }
}

export function DecisionGraph({ nodes, selectedNode, onSelectNode }) {
  const devIdx   = NODE_DEFS.findIndex(n => n.id === 'developer')
  const retryIds = ['executor', 'tester', 'security'].filter(id => {
    const n = nodes[id]
    return n && (n.status === 'retrying' || n.retryCount > 0 || n.status === 'failed')
  })

  return (
    <div style={{ overflowY: 'auto', height: '100%', padding: '6px 0' }}>
      <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{ fontFamily: "'SF Mono','Fira Code','Consolas',monospace", display: 'block' }}>

        {/* ── Connector lines between nodes ──────────────────────────── */}
        {NODE_DEFS.slice(0, -1).map((_, i) => (
          <line key={i}
            x1={CX} y1={cy(i) + NODE_H / 2 + 1}
            x2={CX} y2={cy(i + 1) - NODE_H / 2 - 1}
            stroke="#132035" strokeWidth="2" strokeDasharray="4 3"
          />
        ))}

        {/* ── Retry arcs ──────────────────────────────────────────────── */}
        {retryIds.map(id => {
          const fromI  = NODE_DEFS.findIndex(n => n.id === id)
          const fromCy = cy(fromI)
          const toCy   = cy(devIdx)
          const ns     = nodes[id]
          const active = ns.status === 'retrying'
          const color  = active ? '#D97706' : '#B91C1C'
          const cpx    = NR + 44          // control point outside SVG_W but fine for viewBox

          return (
            <g key={id}>
              {/* Curved path */}
              <path
                d={`M ${NR},${fromCy} C ${cpx},${fromCy} ${cpx},${toCy} ${NR},${toCy}`}
                fill="none" stroke={color} strokeWidth="1.5"
                strokeDasharray={active ? '6 3' : '0'} opacity="0.9"
              >
                {active && (
                  <animate attributeName="stroke-dashoffset"
                    from="18" to="0" dur="0.85s" repeatCount="indefinite" />
                )}
              </path>
              {/* Arrow into developer node */}
              <polygon
                points={`${NR},${toCy} ${NR + 10},${toCy - 5} ${NR + 10},${toCy + 5}`}
                fill={color} opacity="0.9"
              />
              {/* Why-label on arc midpoint */}
              <text x={cpx + 5} y={(fromCy + toCy) / 2}
                fill={color} fontSize="9" fontStyle="italic" dominantBaseline="middle"
              >
                {ns.error ? ns.error.slice(0, 20) : 'retry'}
              </text>
            </g>
          )
        })}

        {/* ── Nodes ──────────────────────────────────────────────────── */}
        {NODE_DEFS.map((def, i) => {
          const ns       = nodes[def.id] || { status: 'idle', llmCalls: [], decisions: [], data: {} }
          const s        = STATUS_STYLE[ns.status] || STATUS_STYLE.idle
          const centerY  = cy(i)
          const nx       = NX
          const ny       = centerY - NODE_H / 2
          const isActive = ns.status === 'running'
          const isSel    = selectedNode === def.id
          const llmN     = ns.llmCalls?.length || 0
          const hasDec   = ns.decisions?.length > 0

          return (
            <g key={def.id} onClick={() => onSelectNode(def.id)}
              style={{ cursor: 'pointer' }}>

              {/* Outer glow for running node */}
              {isActive && (
                <rect x={nx - 3} y={ny - 3} width={NODE_W + 6} height={NODE_H + 6}
                  rx="13" fill="none" stroke="#3B82F6" strokeWidth="1">
                  <animate attributeName="opacity" values="0.7;0.15;0.7" dur="1.3s" repeatCount="indefinite" />
                </rect>
              )}

              {/* Main card */}
              <rect x={nx} y={ny} width={NODE_W} height={NODE_H}
                rx="10" fill={s.fill}
                stroke={isSel ? '#818CF8' : s.stroke}
                strokeWidth={isSel ? 2 : 1.5}
              />

              {/* Left accent stripe */}
              <rect x={nx} y={ny} width={5} height={NODE_H}
                rx="10" fill={s.stroke} opacity="0.9" />

              {/* Icon */}
              <text x={nx + 20} y={centerY + 1} fontSize="20" dominantBaseline="middle">{def.icon}</text>

              {/* Label */}
              <text x={nx + 50} y={centerY - 10}
                fill={s.labelColor} fontSize="13" fontWeight="700" dominantBaseline="middle"
              >{def.label}</text>

              {/* Sub-label */}
              <text x={nx + 50} y={centerY + 10}
                fill={ns.status === 'idle' ? '#2A3F55' : (ns.status === 'running' ? '#60A5FA' : s.subColor)}
                fontSize="10.5" dominantBaseline="middle"
              >{subLabel({ ...ns, id: def.id })}</text>

              {/* Duration badge */}
              {ns.startTime && ns.endTime && (
                <text x={nx + NODE_W - 8} y={ny + NODE_H - 7}
                  fill="#2A4060" fontSize="9" textAnchor="end"
                >{duration(ns.startTime, ns.endTime)}</text>
              )}

              {/* LLM call count badge */}
              {llmN > 0 && (
                <g transform={`translate(${nx + NODE_W - 30}, ${ny + 8})`}>
                  <rect x="0" y="0" width="24" height="16" rx="8"
                    fill="#0D2055" stroke="#2A5090" strokeWidth="1" />
                  <text x="12" y="8.5" fill="#93C5FD" fontSize="9"
                    textAnchor="middle" dominantBaseline="middle" fontWeight="bold">
                    {llmN}
                  </text>
                </g>
              )}

              {/* Purple decision dot */}
              {hasDec && (
                <circle cx={nx + NODE_W - 10} cy={ny + NODE_H - 10}
                  r="4" fill="#7C3AED" stroke="#A78BFA" strokeWidth="1" />
              )}
            </g>
          )
        })}

        {/* ── Legend ────────────────────────────────────────────────── */}
        <g transform={`translate(${NX}, ${SVG_H - 24})`}>
          {[
            ['idle', 'Waiting'], ['running', 'Running'], ['completed', 'Done'],
            ['failed', 'Failed'], ['retrying', 'Retrying'], ['skipped', 'Skipped'],
          ].map(([status, label], i) => (
            <g key={status} transform={`translate(${i * 65}, 0)`}>
              <rect width="10" height="10" rx="2"
                fill={STATUS_STYLE[status].fill}
                stroke={STATUS_STYLE[status].stroke}
                strokeWidth="1.5" />
              <text x="14" y="9" fill="#2A4060" fontSize="9">{label}</text>
            </g>
          ))}
        </g>

        {/* LLM / Decision key */}
        <g transform={`translate(${NX}, ${SVG_H - 8})`}>
          <rect width="16" height="10" rx="5" fill="#0D2055" stroke="#2A5090" strokeWidth="1" />
          <text x="8" y="8" fill="#93C5FD" fontSize="7" textAnchor="middle" dominantBaseline="middle">N</text>
          <text x="20" y="8" fill="#2A4060" fontSize="9">LLM calls</text>
          <circle cx="82" cy="5" r="4" fill="#7C3AED" stroke="#A78BFA" strokeWidth="1" />
          <text x="90" y="8" fill="#2A4060" fontSize="9">Decision</text>
        </g>
      </svg>
    </div>
  )
}
