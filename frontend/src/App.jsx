import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, ArrowRight, Check, ChevronDown, CircleHelp, Clock3, FileImage,
  FlaskConical, HeartPulse, ImagePlus, Info, LoaderCircle, Moon, ShieldCheck,
  Sparkles, Sun, UploadCloud, X, Zap
} from 'lucide-react'

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const classes = ['Benign', 'Malignant', 'Normal']

export default function App() {
  const [theme, setTheme] = useState('dark')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const inputRef = useRef(null)

  useEffect(() => {
    if (!API_URL) { setApiStatus('not-configured'); return }
    let active = true
    fetch(`${API_URL}/health`).then(r => r.json()).then(d => {
      if (active) setApiStatus(d.model_loaded ? 'online' : 'model-missing')
    }).catch(() => { if (active) setApiStatus('offline') })
    return () => { active = false }
  }, [])

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview) }
  }, [preview])

  const statusLabel = {
    checking: 'Checking API', online: 'API connected', 'model-missing': 'Model missing',
    offline: 'API unreachable', 'not-configured': 'API URL not set'
  }[apiStatus]

  function acceptFile(next) {
    if (!next) return
    setError('')
    setResult(null)
    if (!next.type.startsWith('image/')) { setError('Please choose an image file (JPG or PNG).'); return }
    if (next.size > 15 * 1024 * 1024) { setError('Choose an image smaller than 15 MB.'); return }
    if (preview) URL.revokeObjectURL(preview)
    setFile(next)
    setPreview(URL.createObjectURL(next))
  }

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null); setPreview(''); setResult(null); setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function analyze() {
    if (!file) { setError('Upload an image first.'); return }
    if (!API_URL) { setError('Backend URL is not configured. Set VITE_API_URL in Vercel or frontend/.env.local.'); return }
    setBusy(true); setError(''); setResult(null)
    try {
      const body = new FormData()
      body.append('file', file)
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.detail || 'The analysis request failed.')
      setResult(data)
    } catch (e) {
      setError(e.message || 'Could not connect to the API.')
    } finally { setBusy(false) }
  }

  const confidencePct = result ? Math.round(Number(result.confidence) * 100) : 0
  const resultTone = result?.predicted_class === 'Malignant' ? 'warning' : 'positive'

  return <div className={`app ${theme}`}>
    <div className="ambient ambient-a" /><div className="ambient ambient-b" />
    <header className="topbar">
      <a className="brand" href="#" aria-label="PulmoAI home">
        <span className="brand-mark"><Activity size={22} strokeWidth={2.4}/></span>
        <span><b>Pulmo<span>AI</span></b><small>IMAGING LAB</small></span>
      </a>
      <nav className="nav-links"><a className="active" href="#workspace">Workspace</a><a href="#workflow">How it works</a><a href="#about">About</a></nav>
      <div className="top-actions">
        <span className={`api-pill ${apiStatus}`}><i />{statusLabel}</span>
        <button className="icon-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme">{theme === 'dark' ? <Sun size={18}/> : <Moon size={18}/>}</button>
      </div>
    </header>

    <main>
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-dot"/><Sparkles size={14}/> AI-ASSISTED IMAGE EXPLORATION</div>
          <h1>See deeper.<br/><span>Explore smarter.</span></h1>
          <p className="hero-text">An experimental research workspace for exploring lung image classification with a convolutional neural network.</p>
          <div className="hero-points"><span><ShieldCheck size={16}/> Research-first design</span><span><Zap size={16}/> CNN-powered API</span></div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="scan-ring ring-one"/><div className="scan-ring ring-two"/><div className="scan-core"><HeartPulse size={60} strokeWidth={1.15}/><span className="scan-cross cross-a"/><span className="scan-cross cross-b"/></div>
          <div className="float-chip chip-top"><span className="tiny-pulse"/> Image workspace</div>
          <div className="float-chip chip-bottom"><Activity size={15}/> Three-class model</div>
        </div>
      </section>

      <section className="stats-grid" aria-label="Model information">
        <div className="stat-card"><span className="stat-icon mint"><FlaskConical size={19}/></span><div><small>MODEL TYPE</small><b>Convolutional Neural Network</b><span className="stat-sub">3-class image classifier</span></div></div>
        <div className="stat-card"><span className="stat-icon blue"><FileImage size={19}/></span><div><small>INPUT FORMAT</small><b>JPG / PNG</b><span className="stat-sub">Maximum file size 15 MB</span></div></div>
        <div className="stat-card"><span className="stat-icon violet"><Activity size={19}/></span><div><small>REPORTED TEST ACCURACY</small><b>61.45%</b><span className="stat-sub">Experimental · not clinically validated</span></div></div>
      </section>

      <section className="workspace" id="workspace">
        <div className="section-heading"><div><div className="eyebrow">THE WORKSPACE</div><h2>Image analysis</h2><p>Upload an image to request a prediction from your deployed model.</p></div><span className="step-label"><span>01</span> INPUT <ArrowRight size={13}/> <span>02</span> ANALYSIS</span></div>
        <div className="work-grid">
          <div className="panel upload-panel">
            <div className="panel-heading"><div className="panel-title-icon"><ImagePlus size={18}/></div><div><h3>Image input</h3><p>Choose an image for experimental analysis</p></div>{file && <button className="clear-button" onClick={clearFile}><X size={14}/> Clear</button>}</div>
            {!preview ? <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => {e.preventDefault();setDragging(true)}} onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault();setDragging(false);acceptFile(e.dataTransfer.files?.[0])}} onClick={() => inputRef.current?.click()} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}>
              <input ref={inputRef} type="file" accept="image/*" hidden onChange={e => acceptFile(e.target.files?.[0])}/>
              <div className="upload-icon"><UploadCloud size={27}/></div><h4>Drop an image here</h4><p>or click to browse your device</p><button className="browse-button" type="button">Browse files <ArrowRight size={15}/></button><small>IMAGE FILES · JPG, JPEG, PNG</small>
            </div> : <div className="preview-wrap"><img src={preview} alt="Selected image preview"/><div className="preview-overlay"><span><Check size={14}/> Image ready</span><button onClick={() => inputRef.current?.click()}>Replace</button><input ref={inputRef} type="file" accept="image/*" hidden onChange={e => acceptFile(e.target.files?.[0])}/></div><div className="file-meta"><FileImage size={15}/><span>{file?.name}</span><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></div></div>}
            <div className="upload-foot"><Info size={15}/><span>Use images you have permission to share. Uploads are sent to the configured backend for inference.</span></div>
          </div>

          <div className="panel result-panel">
            <div className="panel-heading"><div className="panel-title-icon purple"><Activity size={18}/></div><div><h3>Analysis result</h3><p>Model response appears here</p></div><span className="live-tag"><i/> LIVE API</span></div>
            {!result && !busy && <div className="empty-result"><div className="result-orbit"><div><Activity size={29}/></div><span className="orbit-dot dot-1"/><span className="orbit-dot dot-2"/><span className="orbit-dot dot-3"/></div><h4>Ready when you are</h4><p>Your model output and class scores will appear here after you submit an image.</p><div className="class-chips">{classes.map(c => <span key={c}>{c}</span>)}</div></div>}
            {busy && <div className="loading-result"><LoaderCircle size={34} className="spin"/><h4>Requesting model analysis</h4><p>Sending your image to the configured API…</p></div>}
            {result && <div className="result-content"><div className={`prediction-banner ${resultTone}`}><span className="prediction-symbol">{resultTone === 'warning' ? <Info size={23}/> : <Check size={23}/>}</span><div><small>MODEL PREDICTION</small><h4>{result.predicted_class}</h4><p>Experimental classification output</p></div><div className="confidence"><b>{confidencePct}%</b><small>top score</small></div></div><div className="score-heading"><b>Class scores</b><span>Model output</span></div><div className="score-list">{classes.map(c => { const score = Number(result.scores?.[c] ?? 0); return <div className="score-row" key={c}><div><span>{c}</span><b>{Math.round(score * 100)}%</b></div><div className="score-track"><i style={{width:`${Math.max(0,Math.min(100,score*100))}%`}}/></div></div>})}</div><div className="result-disclaimer"><Info size={15}/>{result.notice || 'Experimental output only; not a medical diagnosis.'}</div></div>}
            {error && <div className="error-box"><CircleHelp size={17}/><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={14}/></button></div>}
            <button className="analyze-button" disabled={!file || busy} onClick={analyze}>{busy ? <><LoaderCircle size={17} className="spin"/> Analyzing…</> : <>Analyze image <ArrowRight size={17}/></>}</button>
            <p className="api-note">{API_URL ? <>API endpoint <code>{API_URL}</code></> : <>Set <code>VITE_API_URL</code> to connect your Render API</>}</p>
          </div>
        </div>
      </section>

      <section className="workflow" id="workflow">
        <div className="section-heading compact"><div><div className="eyebrow">THE PIPELINE</div><h2>From image to output</h2></div><span className="muted-note"><Clock3 size={15}/> Simple 3-step workflow</span></div>
        <div className="workflow-grid"><div className="flow-card"><span className="flow-num">01</span><div className="flow-icon"><UploadCloud size={21}/></div><h3>Upload image</h3><p>Select a JPG or PNG image from your device. The browser validates the file before sending.</p></div><div className="flow-connector"><ArrowRight size={18}/></div><div className="flow-card"><span className="flow-num">02</span><div className="flow-icon"><FlaskConical size={21}/></div><h3>Model inference</h3><p>The Render API resizes the image and passes it to the saved Keras model.</p></div><div className="flow-connector"><ArrowRight size={18}/></div><div className="flow-card"><span className="flow-num">03</span><div className="flow-icon"><Activity size={21}/></div><h3>View class scores</h3><p>See the predicted class and scores returned by the model API.</p></div></div>
      </section>

      <section className="notice-panel" id="about"><div className="notice-icon"><ShieldCheck size={21}/></div><div><h3>Research prototype — not a medical device</h3><p>This student project is for educational and research exploration only. Its reported test accuracy is 61.45%, and its predictions are not clinically validated. Do not use it to diagnose, rule out, or make treatment decisions about cancer. Consult a qualified medical professional for health concerns.</p></div><ChevronDown size={17} className="notice-chevron"/></section>
    </main>
    <footer><a className="footer-brand" href="#"><span className="brand-mark small"><Activity size={16}/></span> PulmoAI <span>IMAGING LAB</span></a><p>Built for academic research · Vercel frontend + Render API</p><span className="footer-status"><i/> Experimental build</span></footer>
  </div>
}
