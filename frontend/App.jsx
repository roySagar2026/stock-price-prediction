import { useState, useEffect, useCallback, useRef } from "react";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ─── INLINE STYLES / DESIGN TOKENS ───────────────────────────────────────────
const T = {
  bg:      "#070B14",
  surface: "#0D1421",
  card:    "#111827",
  border:  "#1E2D45",
  accent:  "#00D4FF",
  green:   "#00FFA3",
  red:     "#FF4D6A",
  amber:   "#FFB340",
  muted:   "#4A5568",
  text:    "#E2E8F0",
  sub:     "#8892A4",
  font:    "'Syne', sans-serif",
  mono:    "'JetBrains Mono', monospace",
};

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function fmt(v, d=2) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return Number(v).toFixed(d);
}
function pct(v) { return v >= 0 ? `+${fmt(v)}%` : `${fmt(v)}%`; }
function clr(v) { return v >= 0 ? T.green : T.red; }

// ─── MINI SPARKLINE ───────────────────────────────────────────────────────────
function Sparkline({ data = [], width = 180, height = 44, color = T.accent }) {
  if (!data || data.length < 2) return null;
  const mn = Math.min(...data), mx = Math.max(...data), rng = mx - mn || 1;
  const pts = data.map((v, i) =>
    `${(i / (data.length - 1)) * width},${height - ((v - mn) / rng) * (height - 4) - 2}`
  ).join(" ");
  const areaBottom = `${width},${height} 0,${height}`;
  return (
    <svg width={width} height={height} style={{ display:"block" }}>
      <defs>
        <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3"/>
          <stop offset="100%" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      <polygon points={`${pts} ${areaBottom}`} fill="url(#sg)" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round"/>
    </svg>
  );
}

// ─── PRICE CHART ─────────────────────────────────────────────────────────────
function PriceChart({ data = [] }) {
  if (!data.length) return null;
  const W = 680, H = 200, PAD = { t:16, r:16, b:32, l:56 };
  const closes = data.map(d => d.close).filter(Boolean);
  const mn = Math.min(...closes), mx = Math.max(...closes), rng = mx - mn || 1;
  const xScale = i => PAD.l + (i / (data.length - 1)) * (W - PAD.l - PAD.r);
  const yScale = v => PAD.t + (1 - (v - mn) / rng) * (H - PAD.t - PAD.b);

  const linePts = data.map((d, i) => `${xScale(i)},${yScale(d.close)}`).join(" ");
  const areaPts = `${linePts} ${xScale(data.length-1)},${H-PAD.b} ${xScale(0)},${H-PAD.b}`;

  // Y-axis labels
  const yTicks = [mn, mn + rng * 0.25, mn + rng * 0.5, mn + rng * 0.75, mx];
  // X-axis labels (every ~30 days)
  const xLabels = [];
  const step = Math.max(1, Math.floor(data.length / 6));
  for (let i = 0; i < data.length; i += step) {
    xLabels.push({ i, label: data[i].date.slice(0,7) });
  }

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow:"visible" }}>
      <defs>
        <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={T.accent} stopOpacity="0.2"/>
          <stop offset="100%" stopColor={T.accent} stopOpacity="0"/>
        </linearGradient>
      </defs>
      {/* Grid */}
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={PAD.l} y1={yScale(v)} x2={W-PAD.r} y2={yScale(v)}
            stroke={T.border} strokeDasharray="3 3" strokeWidth="0.5"/>
          <text x={PAD.l-6} y={yScale(v)+4} textAnchor="end"
            fill={T.muted} fontSize="9" fontFamily={T.mono}>{fmt(v,0)}</text>
        </g>
      ))}
      {/* Area fill */}
      <polygon points={areaPts} fill="url(#cg)"/>
      {/* Line */}
      <polyline points={linePts} fill="none" stroke={T.accent}
        strokeWidth="1.5" strokeLinejoin="round"/>
      {/* X labels */}
      {xLabels.map(({ i, label }) => (
        <text key={i} x={xScale(i)} y={H - 6} textAnchor="middle"
          fill={T.muted} fontSize="8" fontFamily={T.mono}>{label}</text>
      ))}
    </svg>
  );
}

// ─── RSI CHART ────────────────────────────────────────────────────────────────
function RSIChart({ data = [] }) {
  if (!data.length) return null;
  const W = 680, H = 80, PAD = { t:8, r:16, b:24, l:56 };
  const xScale = i => PAD.l + (i / (data.length - 1)) * (W - PAD.l - PAD.r);
  const yScale = v => PAD.t + (1 - v / 100) * (H - PAD.t - PAD.b);
  const validData = data.filter(d => d.rsi_14 !== null);
  if (!validData.length) return null;
  const linePts = data.map((d, i) =>
    d.rsi_14 !== null ? `${xScale(i)},${yScale(d.rsi_14)}` : null
  ).filter(Boolean).join(" ");

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
      <text x={PAD.l-4} y={PAD.t+10} textAnchor="end" fill={T.muted} fontSize="8" fontFamily={T.mono}>RSI</text>
      <line x1={PAD.l} y1={yScale(70)} x2={W-PAD.r} y2={yScale(70)} stroke={T.red} strokeDasharray="3 2" strokeWidth="0.5" strokeOpacity="0.6"/>
      <line x1={PAD.l} y1={yScale(30)} x2={W-PAD.r} y2={yScale(30)} stroke={T.green} strokeDasharray="3 2" strokeWidth="0.5" strokeOpacity="0.6"/>
      <polyline points={linePts} fill="none" stroke={T.amber} strokeWidth="1.2" strokeLinejoin="round"/>
      {[30,50,70].map(v => (
        <text key={v} x={PAD.l-6} y={yScale(v)+4} textAnchor="end" fill={T.muted} fontSize="7" fontFamily={T.mono}>{v}</text>
      ))}
    </svg>
  );
}

// ─── CONFIDENCE GAUGE ────────────────────────────────────────────────────────
function ConfidenceGauge({ probability = 0.5, direction = "UP" }) {
  const pct_val = Math.round(probability * 100);
  const isUp = direction === "UP";
  const color = isUp ? T.green : T.red;
  const angle = -90 + (probability * 180);
  const r = 60, cx = 80, cy = 80;
  const bgArc = describeArc(cx, cy, r, -90, 90);
  const valArc = describeArc(cx, cy, r, -90, angle);
  return (
    <svg width="160" height="100" viewBox="0 0 160 100">
      <path d={bgArc} fill="none" stroke={T.border} strokeWidth="10" strokeLinecap="round"/>
      <path d={valArc} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"/>
      <text x={cx} y={cy-8} textAnchor="middle" fill={color}
        fontSize="26" fontFamily={T.font} fontWeight="700">{pct_val}%</text>
      <text x={cx} y={cy+10} textAnchor="middle" fill={T.sub}
        fontSize="10" fontFamily={T.font}>{isUp ? "↑ BULLISH" : "↓ BEARISH"}</text>
    </svg>
  );
}
function describeArc(x, y, r, startAngle, endAngle) {
  const a2r = a => a * Math.PI / 180;
  const sx = x + r * Math.cos(a2r(startAngle));
  const sy = y + r * Math.sin(a2r(startAngle));
  const ex = x + r * Math.cos(a2r(endAngle));
  const ey = y + r * Math.sin(a2r(endAngle));
  const largeArc = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${sx} ${sy} A ${r} ${r} 0 ${largeArc} 1 ${ex} ${ey}`;
}

// ─── FEATURE BAR ─────────────────────────────────────────────────────────────
function FeatureBar({ name, importance, max }) {
  const w = max > 0 ? (importance / max) * 100 : 0;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
        <span style={{ color:T.sub, fontSize:11, fontFamily:T.mono }}>{name}</span>
        <span style={{ color:T.accent, fontSize:11, fontFamily:T.mono }}>{importance.toFixed(1)}</span>
      </div>
      <div style={{ height:4, background:T.border, borderRadius:2 }}>
        <div style={{ height:"100%", width:`${w}%`, background:T.accent,
          borderRadius:2, transition:"width 0.8s ease" }}/>
      </div>
    </div>
  );
}

// ─── STAT CARD ────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, color = T.text }) {
  return (
    <div style={{
      background: T.card, border:`1px solid ${T.border}`,
      borderRadius: 10, padding:"12px 16px",
    }}>
      <div style={{ color:T.sub, fontSize:10, textTransform:"uppercase",
        letterSpacing:"0.1em", fontFamily:T.font, marginBottom:4 }}>{label}</div>
      <div style={{ color, fontSize:20, fontWeight:700, fontFamily:T.font, lineHeight:1.1 }}>{value}</div>
      {sub && <div style={{ color:T.muted, fontSize:11, marginTop:3, fontFamily:T.mono }}>{sub}</div>}
    </div>
  );
}

// ─── SIGNAL BADGE ────────────────────────────────────────────────────────────
function SignalBadge({ signal }) {
  const map = {
    BUY:  { bg:"#00FFA318", border:"#00FFA360", text:T.green },
    SELL: { bg:"#FF4D6A18", border:"#FF4D6A60", text:T.red   },
    HOLD: { bg:"#FFB34018", border:"#FFB34060", text:T.amber },
  };
  const s = map[signal] || map.HOLD;
  return (
    <div style={{
      display:"inline-flex", alignItems:"center", gap:6,
      padding:"6px 16px", borderRadius:20,
      background: s.bg, border:`1px solid ${s.border}`,
    }}>
      <div style={{ width:6, height:6, borderRadius:"50%",
        background:s.text, boxShadow:`0 0 8px ${s.text}` }}/>
      <span style={{ color:s.text, fontWeight:700, fontSize:13,
        letterSpacing:"0.1em", fontFamily:T.font }}>{signal}</span>
    </div>
  );
}

// ─── SEARCH BAR ───────────────────────────────────────────────────────────────
function SearchBar({ onSearch, loading }) {
  const [q, setQ] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [showSug, setShowSug] = useState(false);
  const sugRef = useRef(null);
  const timerRef = useRef(null);

  const fetchSuggestions = useCallback(async (val) => {
    if (!val || val.length < 2) { setSuggestions([]); return; }
    try {
      const r = await fetch(`${API_BASE}/search?q=${encodeURIComponent(val)}&limit=6`);
      const j = await r.json();
      setSuggestions(j.results || []);
    } catch { setSuggestions([]); }
  }, []);

  const handleChange = (e) => {
    const val = e.target.value;
    setQ(val);
    setShowSug(true);
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => fetchSuggestions(val), 300);
  };

  const handleSelect = (item) => {
    setQ(item.name || item.ticker);
    setSuggestions([]);
    setShowSug(false);
    onSearch(item.ticker);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!q.trim()) return;
    setSuggestions([]);
    setShowSug(false);
    onSearch(q.trim());
  };

  return (
    <div style={{ position:"relative", maxWidth:560, margin:"0 auto" }}>
      <form onSubmit={handleSubmit}>
        <div style={{ display:"flex", gap:8, alignItems:"center" }}>
          <div style={{ flex:1, position:"relative" }}>
            <span style={{
              position:"absolute", left:14, top:"50%", transform:"translateY(-50%)",
              color:T.muted, fontSize:16,
            }}>⌕</span>
            <input
              value={q}
              onChange={handleChange}
              onFocus={() => setShowSug(true)}
              onBlur={() => setTimeout(() => setShowSug(false), 180)}
              placeholder="Apple, TSLA, Reliance, NVDA…"
              style={{
                width:"100%", boxSizing:"border-box",
                background: T.surface, border:`1px solid ${T.border}`,
                borderRadius:10, padding:"12px 14px 12px 40px",
                color:T.text, fontSize:15, fontFamily:T.font,
                outline:"none",
                transition:"border-color 0.2s",
              }}
              onMouseEnter={e => e.target.style.borderColor = T.accent}
              onMouseLeave={e => e.target.style.borderColor = T.border}
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            style={{
              background: loading ? T.muted : T.accent,
              color: T.bg, border:"none", borderRadius:10,
              padding:"12px 22px", fontWeight:700, fontSize:14,
              cursor: loading ? "default" : "pointer",
              fontFamily:T.font, letterSpacing:"0.05em",
              transition:"background 0.2s",
              whiteSpace:"nowrap",
            }}
          >
            {loading ? "ANALYZING…" : "PREDICT →"}
          </button>
        </div>
      </form>

      {/* Suggestions dropdown */}
      {showSug && suggestions.length > 0 && (
        <div ref={sugRef} style={{
          position:"absolute", top:"100%", left:0, right:60,
          background:T.card, border:`1px solid ${T.border}`,
          borderRadius:10, marginTop:4, zIndex:100, overflow:"hidden",
        }}>
          {suggestions.map((s, i) => (
            <div key={i}
              onMouseDown={() => handleSelect(s)}
              style={{
                padding:"10px 14px", cursor:"pointer",
                display:"flex", alignItems:"center", gap:10,
                borderBottom: i < suggestions.length-1 ? `1px solid ${T.border}` : "none",
                transition:"background 0.15s",
              }}
              onMouseEnter={e => e.currentTarget.style.background = T.surface}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}
            >
              <span style={{ color:T.accent, fontFamily:T.mono, fontSize:12, minWidth:80 }}>{s.ticker}</span>
              <span style={{ color:T.text, fontSize:13, fontFamily:T.font }}>{s.name}</span>
              <span style={{ color:T.muted, fontSize:10, marginLeft:"auto", fontFamily:T.mono }}>{s.exchange}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── LOADING SKELETON ────────────────────────────────────────────────────────
function LoadingPulse() {
  const boxStyle = (w, h, mb=12) => ({
    width: w, height: h, borderRadius:8, marginBottom:mb,
    background:`linear-gradient(90deg, ${T.surface} 0%, ${T.border} 50%, ${T.surface} 100%)`,
    backgroundSize:"200% 100%",
    animation:"shimmer 1.5s infinite",
  });
  return (
    <div style={{ marginTop:32 }}>
      <style>{`@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}`}</style>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:12, marginBottom:16 }}>
        {[1,2,3,4].map(i=><div key={i} style={boxStyle("100%",76,0)}/>)}
      </div>
      <div style={boxStyle("100%",260)}/>
      <div style={boxStyle("100%",100)}/>
      <div style={{ display:"grid", gridTemplateColumns:"160px 1fr", gap:16, marginTop:16 }}>
        <div style={boxStyle("100%",100,0)}/>
        <div style={boxStyle("100%",100,0)}/>
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function StockPulse() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(null);
  const [activeTab, setActiveTab] = useState("chart");

  const handleSearch = async (query) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setLatency(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ query }),
      });
      const json = await res.json();
      if (!json.ok) {
        setError(json.error || "Prediction failed.");
      } else {
        setResult(json.data);
        setLatency(json.latency_ms);
      }
    } catch (e) {
      setError(`Cannot connect to API at ${API_BASE}. Start the backend with: uvicorn api.app:app --reload`);
    } finally {
      setLoading(false);
    }
  };

  const d = result;
  const chartData = d?.chart_data || [];
  const closes = chartData.map(r => r.close).filter(Boolean);
  const maxImp = d?.top_features?.length ? Math.max(...d.top_features.map(f=>f.importance)) : 1;

  return (
    <div style={{
      minHeight:"100vh", background:T.bg, color:T.text,
      fontFamily:T.font, padding:"0 0 64px",
    }}>
      {/* Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width:4px } ::-webkit-scrollbar-track { background:${T.bg} }
        ::-webkit-scrollbar-thumb { background:${T.border}; border-radius:2px }
        input { outline:none; }
        button { transition: opacity 0.15s, transform 0.1s; }
        button:active { transform: scale(0.97); }
      `}</style>

      {/* ── HERO HEADER ── */}
      <div style={{
        borderBottom:`1px solid ${T.border}`,
        padding:"28px 32px 24px",
        display:"flex", alignItems:"center", justifyContent:"space-between",
      }}>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <div style={{
            width:34, height:34, borderRadius:8,
            background:`linear-gradient(135deg, ${T.accent}, #0088AA)`,
            display:"flex", alignItems:"center", justifyContent:"center",
            fontSize:16,
          }}>📈</div>
          <div>
            <div style={{ fontSize:18, fontWeight:800, letterSpacing:"-0.02em",
              color:T.text }}>StockPulse</div>
            <div style={{ fontSize:10, color:T.muted, letterSpacing:"0.1em",
              textTransform:"uppercase", fontFamily:T.mono }}>
              Global Large-Cap ML Prediction
            </div>
          </div>
        </div>
        <div style={{ display:"flex", gap:24, fontSize:11, color:T.muted,
          fontFamily:T.mono, letterSpacing:"0.05em" }}>
          {["USA · India · China · Europe","LSTM + LightGBM + XGBoost","5-Day Forward"].map((s,i)=>(
            <span key={i}>• {s}</span>
          ))}
        </div>
      </div>

      {/* ── SEARCH SECTION ── */}
      <div style={{
        padding:"36px 32px 28px",
        background:`radial-gradient(ellipse at 50% 0%, #00D4FF0A 0%, transparent 70%)`,
        borderBottom:`1px solid ${T.border}`,
      }}>
        <div style={{ textAlign:"center", marginBottom:24 }}>
          <h1 style={{
            fontSize:34, fontWeight:800, margin:0,
            letterSpacing:"-0.03em", lineHeight:1.1,
            background:`linear-gradient(90deg, ${T.text}, ${T.accent})`,
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent",
          }}>
            Predict Stock Movement
          </h1>
          <p style={{ color:T.sub, margin:"8px 0 0", fontSize:13 }}>
            Enter any company name or ticker • Dynamic resolution • Real-time data
          </p>
        </div>
        <SearchBar onSearch={handleSearch} loading={loading}/>

        {/* Quick picks */}
        <div style={{ display:"flex", gap:8, justifyContent:"center", marginTop:16, flexWrap:"wrap" }}>
          {["Apple","NVDA","Tesla","HDFC Bank","RELIANCE.NS","0700.HK","ASML"].map(s=>(
            <button key={s} onClick={() => handleSearch(s)}
              style={{
                background:"transparent", border:`1px solid ${T.border}`,
                borderRadius:20, padding:"4px 14px", color:T.sub, fontSize:11,
                cursor:"pointer", fontFamily:T.mono, letterSpacing:"0.04em",
              }}
              onMouseEnter={e=>{ e.currentTarget.style.borderColor=T.accent; e.currentTarget.style.color=T.accent; }}
              onMouseLeave={e=>{ e.currentTarget.style.borderColor=T.border; e.currentTarget.style.color=T.sub; }}
            >{s}</button>
          ))}
        </div>
      </div>

      {/* ── CONTENT ── */}
      <div style={{ maxWidth:900, margin:"0 auto", padding:"0 32px" }}>

        {/* Loading */}
        {loading && <LoadingPulse />}

        {/* Error */}
        {error && (
          <div style={{
            marginTop:32, padding:"16px 20px", borderRadius:10,
            background:"#FF4D6A10", border:`1px solid #FF4D6A40`,
            color: T.red, fontFamily:T.mono, fontSize:13, lineHeight:1.6,
          }}>
            ⚠ {error}
          </div>
        )}

        {/* Results */}
        {d && !loading && (
          <div style={{ marginTop:32, animation:"fadeIn 0.4s ease" }}>
            <style>{`@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}`}</style>

            {/* ── Company header ── */}
            <div style={{
              display:"flex", alignItems:"flex-start", justifyContent:"space-between",
              marginBottom:20, gap:16, flexWrap:"wrap",
            }}>
              <div>
                <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:6 }}>
                  <span style={{ fontFamily:T.mono, fontSize:13, color:T.accent,
                    background:"#00D4FF15", padding:"3px 10px", borderRadius:6 }}>
                    {d.ticker}
                  </span>
                  <span style={{ color:T.muted, fontSize:11, fontFamily:T.mono }}>
                    {d.market} · {d.sector}
                  </span>
                  <span style={{ color:T.muted, fontSize:11, fontFamily:T.mono }}>
                    {d.currency}
                  </span>
                </div>
                <h2 style={{ margin:0, fontSize:22, fontWeight:800,
                  letterSpacing:"-0.02em", color:T.text }}>
                  {d.company_name}
                </h2>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:12 }}>
                <SignalBadge signal={d.signal}/>
                {latency && (
                  <span style={{ color:T.muted, fontSize:10, fontFamily:T.mono }}>
                    {latency}ms
                  </span>
                )}
              </div>
            </div>

            {/* ── Stat grid ── */}
            <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:10, marginBottom:16 }}>
              <StatCard label="Last Close"
                value={`${d.currency === "USD" ? "$" : ""}${fmt(d.last_close, d.last_close > 100 ? 2 : 4)}`}
                sub={`1D: ${pct(d.price_change_1d)}`}
                color={clr(d.price_change_1d)}/>
              <StatCard label="5-Day Forecast"
                value={d.direction === "UP" ? "▲ UP" : "▼ DOWN"}
                sub={`P(UP) = ${fmt(d.probability*100,1)}%`}
                color={d.direction==="UP" ? T.green : T.red}/>
              <StatCard label="Model AUC"
                value={fmt(d.model_auc,4)}
                sub={d.model_name}
                color={d.model_auc > 0.6 ? T.green : d.model_auc > 0.55 ? T.amber : T.red}/>
              <StatCard label="Confidence"
                value={d.confidence_label}
                sub={`F1: ${fmt(d.model_f1,4)}`}
                color={d.confidence_label==="High" ? T.green : d.confidence_label==="Medium" ? T.amber : T.muted}/>
            </div>

            {/* ── Tabs ── */}
            <div style={{ display:"flex", gap:4, marginBottom:16,
              borderBottom:`1px solid ${T.border}`, paddingBottom:0 }}>
              {[["chart","Price & Indicators"],["analysis","ML Analysis"],["report","Report"]].map(([id,label])=>(
                <button key={id} onClick={()=>setActiveTab(id)}
                  style={{
                    background:"none", border:"none", cursor:"pointer",
                    padding:"8px 18px", fontFamily:T.font, fontSize:12,
                    fontWeight:600, letterSpacing:"0.05em",
                    color: activeTab===id ? T.accent : T.muted,
                    borderBottom: activeTab===id ? `2px solid ${T.accent}` : "2px solid transparent",
                    marginBottom:"-1px", transition:"color 0.2s",
                  }}
                >{label}</button>
              ))}
            </div>

            {/* ── Chart Tab ── */}
            {activeTab==="chart" && (
              <div>
                <div style={{
                  background:T.card, border:`1px solid ${T.border}`,
                  borderRadius:12, padding:"20px", marginBottom:12,
                }}>
                  <div style={{ display:"flex", alignItems:"center",
                    justifyContent:"space-between", marginBottom:12 }}>
                    <span style={{ color:T.sub, fontSize:11, fontFamily:T.font,
                      textTransform:"uppercase", letterSpacing:"0.1em" }}>
                      Price History ({d.data_period_days} trading days)
                    </span>
                    <Sparkline data={closes.slice(-30)} width={80} height={28}
                      color={d.price_change_5d >= 0 ? T.green : T.red}/>
                  </div>
                  <PriceChart data={chartData}/>
                </div>
                <div style={{
                  background:T.card, border:`1px solid ${T.border}`,
                  borderRadius:12, padding:"16px 20px",
                }}>
                  <div style={{ color:T.sub, fontSize:10, textTransform:"uppercase",
                    letterSpacing:"0.1em", marginBottom:8, fontFamily:T.font }}>
                    RSI-14 (Overbought &gt;70 · Oversold &lt;30)
                  </div>
                  <RSIChart data={chartData}/>
                </div>
              </div>
            )}

            {/* ── Analysis Tab ── */}
            {activeTab==="analysis" && (
              <div style={{ display:"grid", gridTemplateColumns:"180px 1fr", gap:16 }}>
                {/* Confidence gauge */}
                <div style={{
                  background:T.card, border:`1px solid ${T.border}`,
                  borderRadius:12, padding:"20px", textAlign:"center",
                }}>
                  <div style={{ color:T.sub, fontSize:10, textTransform:"uppercase",
                    letterSpacing:"0.1em", marginBottom:12, fontFamily:T.font }}>
                    Prediction
                  </div>
                  <ConfidenceGauge probability={d.probability} direction={d.direction}/>
                  <div style={{ marginTop:12 }}>
                    <div style={{ fontSize:11, color:T.muted, fontFamily:T.mono }}>
                      Horizon: {d.horizon_days} trading days
                    </div>
                    <div style={{ fontSize:11, color:T.muted, fontFamily:T.mono, marginTop:4 }}>
                      Threshold: +0.5% = UP
                    </div>
                  </div>
                </div>

                {/* Feature importance */}
                <div style={{
                  background:T.card, border:`1px solid ${T.border}`,
                  borderRadius:12, padding:"20px",
                }}>
                  <div style={{ color:T.sub, fontSize:10, textTransform:"uppercase",
                    letterSpacing:"0.1em", marginBottom:14, fontFamily:T.font }}>
                    Top Features (LightGBM Gain Importance)
                  </div>
                  {d.top_features?.map((f, i) => (
                    <FeatureBar key={i} name={f.name} importance={f.importance} max={maxImp}/>
                  ))}
                </div>
              </div>
            )}

            {/* ── Report Tab ── */}
            {activeTab==="report" && (
              <div style={{
                background:T.card, border:`1px solid ${T.border}`,
                borderRadius:12, padding:"24px",
              }}>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
                  {/* Prediction report */}
                  <div>
                    <div style={{ color:T.accent, fontSize:11, fontFamily:T.mono,
                      textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:12 }}>
                      Prediction Report
                    </div>
                    {[
                      ["Company",       d.company_name],
                      ["Ticker",        d.ticker],
                      ["Market",        d.market],
                      ["Sector",        d.sector],
                      ["Currency",      d.currency],
                      ["Exchange",      d.exchange || "—"],
                      ["Direction",     d.direction],
                      ["Probability",   `${(d.probability*100).toFixed(2)}%`],
                      ["Signal",        d.signal],
                      ["Confidence",    d.confidence_label],
                      ["Horizon",       `${d.horizon_days} trading days`],
                      ["Threshold",     "+0.5% (0.005)"],
                    ].map(([k,v]) => (
                      <div key={k} style={{ display:"flex", justifyContent:"space-between",
                        padding:"6px 0", borderBottom:`1px solid ${T.border}20` }}>
                        <span style={{ color:T.muted, fontSize:12, fontFamily:T.mono }}>{k}</span>
                        <span style={{ color:T.text, fontSize:12, fontFamily:T.mono,
                          fontWeight:600 }}>{v}</span>
                      </div>
                    ))}
                  </div>

                  {/* Model report */}
                  <div>
                    <div style={{ color:T.accent, fontSize:11, fontFamily:T.mono,
                      textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:12 }}>
                      Model Report
                    </div>
                    {[
                      ["Model",         d.model_name],
                      ["Test AUC",      fmt(d.model_auc, 4)],
                      ["Test F1",       fmt(d.model_f1, 4)],
                      ["Data Days",     d.data_period_days],
                      ["Raw Rows",      d.data_quality?.raw_rows],
                      ["Clean Rows",    d.data_quality?.clean_rows],
                      ["Dropped Rows",  d.data_quality?.dropped_rows],
                      ["Missing %",     `${fmt(d.data_quality?.missing_pct_raw, 2)}%`],
                      ["Date From",     d.data_quality?.date_range?.start],
                      ["Date To",       d.data_quality?.date_range?.end],
                      ["Fetched At",    d.fetch_timestamp?.slice(0,19).replace("T"," ")],
                      ["1D Change",     pct(d.price_change_1d)],
                      ["5D Change",     pct(d.price_change_5d)],
                    ].map(([k,v]) => (
                      <div key={k} style={{ display:"flex", justifyContent:"space-between",
                        padding:"6px 0", borderBottom:`1px solid ${T.border}20` }}>
                        <span style={{ color:T.muted, fontSize:12, fontFamily:T.mono }}>{k}</span>
                        <span style={{ color:T.text, fontSize:12, fontFamily:T.mono,
                          fontWeight:600 }}>{v ?? "—"}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Disclaimer */}
                <div style={{
                  marginTop:20, padding:"12px 16px", borderRadius:8,
                  background:"#FFB34010", border:`1px solid #FFB34030`,
                  color:T.amber, fontSize:11, fontFamily:T.mono, lineHeight:1.5,
                }}>
                  ⚠ DISCLAIMER: StockPulse predictions are for educational and research purposes only.
                  They do not constitute financial advice. Past model performance does not guarantee future results.
                  Always conduct your own due diligence before making investment decisions.
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Empty state ── */}
        {!d && !loading && !error && (
          <div style={{
            textAlign:"center", padding:"64px 0", color:T.muted,
          }}>
            <div style={{ fontSize:48, marginBottom:16, opacity:0.3 }}>📊</div>
            <div style={{ fontSize:14, fontFamily:T.font, color:T.sub }}>
              Search for any stock to see the prediction
            </div>
            <div style={{ fontSize:12, fontFamily:T.mono, color:T.muted, marginTop:8 }}>
              Supports 4 markets: US · India · China/HK · Europe
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
