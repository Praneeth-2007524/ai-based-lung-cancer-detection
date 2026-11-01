import { useRef, useState } from 'react';
import './styles.css';

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const MAX_FILE_SIZE = 15 * 1024 * 1024;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

function pct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n >= 0 && n <= 1 ? n * 100 : n));
}
function imageData(value) {
  if (!value) return '';
  return value.startsWith('data:') ? value : `data:image/png;base64,${value}`;
}

export default function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [reportBusy, setReportBusy] = useState(false);
  const [error, setError] = useState('');
  const [activeView, setActiveView] = useState('Original image');
  const [patientId, setPatientId] = useState('DEMO-001');
  const [age, setAge] = useState('45');
  const [gender, setGender] = useState('Other');

  function chooseFile() { inputRef.current?.click(); }

  function handleFile(nextFile) {
    setError('');
    setResult(null);
    if (!nextFile) return;
    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      setError('Please select a JPG, PNG, or WebP image.');
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setError('Please select an image smaller than 15 MB.');
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setActiveView('Original image');
  }

  function clearImage() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview('');
    setResult(null);
    setError('');
    if (inputRef.current) inputRef.current.value = '';
  }

  async function analyzeImage() {
    if (!file) { setError('Choose an image before starting analysis.'); return; }
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body });
      const raw = await response.text();
      let data;
      try { data = JSON.parse(raw); }
      catch { throw new Error(`Invalid server response (HTTP ${response.status}).`); }
      if (!response.ok) throw new Error(data.detail || data.message || `Server error: ${response.status}`);
      const predicted = data.predicted_class || data.prediction || data.class_name || data.predicted_label;
      if (!predicted) throw new Error('The API response did not contain a prediction.');
      setResult({ ...data, predicted_class: predicted });
      setActiveView('Original image');
    } catch (e) {
      setError(`${e.message || 'Could not connect to the backend.'} Check that ${API_URL} is running.`);
    } finally { setBusy(false); }
  }

  async function downloadReport() {
    if (!file || !result) return;
    setReportBusy(true);
    setError('');
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch(`${API_URL}/report`, { method: 'POST', body });
      if (!response.ok) {
        const raw = await response.text();
        let message = raw;
        try { message = JSON.parse(raw).detail || raw; } catch {}
        throw new Error(message || `HTTP ${response.status}`);
      }
      const blob = await response.blob();
      if (blob.type && !blob.type.includes('pdf')) throw new Error('The report endpoint did not return a PDF.');
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'lungsight-analysis-report.pdf';
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setError(`PDF download failed: ${e.message}`); }
    finally { setReportBusy(false); }
  }

  async function shareResult() {
    if (!result) return;
    const text = `LungSight AI research output\nPrediction: ${result.predicted_class}\nRisk category: ${result.risk_category || 'Not provided'}\nModel score: ${pct(result.confidence_percent ?? result.confidence).toFixed(1)}%\nResearch use only — not a medical diagnosis.`;
    try {
      if (navigator.share) await navigator.share({ title: 'LungSight AI result', text });
      else if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        setError('Result summary copied to clipboard.');
      } else setError('Sharing is not supported in this browser.');
    } catch (e) { if (e?.name !== 'AbortError') setError('Unable to share this result.'); }
  }

  const scores = result?.scores && typeof result.scores === 'object' ? Object.entries(result.scores) : [];
  const confidence = pct(result?.confidence_percent ?? result?.confidence);
  const malignant = pct(result?.malignant_score_percent ?? result?.scores?.Malignant ?? 0);
  const risk = result?.risk_category || 'Not provided';
  const views = [
    { label: 'Original image', src: preview },
    { label: 'Grad-CAM heatmap', src: imageData(result?.heatmap_image) },
    { label: 'Attention overlay', src: imageData(result?.overlay_image) }
  ];
  const currentImage = views.find(v => v.label === activeView)?.src || preview;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">✚</div><div><div className="brand-name">LungSight <b>AI</b></div><div className="brand-caption">LUNG ANALYSIS SYSTEM</div></div></div>
        <nav className="navigation">
          <div className="nav-item active"><span>⌂</span>Dashboard</div>
          <button className="nav-item" onClick={chooseFile}><span>♧</span>New Analysis</button>
          <div className="nav-item"><span>▤</span>Patient Records</div>
          <div className="nav-item"><span>▥</span>Analytics</div>
          <div className="nav-item"><span>▧</span>Reports</div>
          <div className="nav-item"><span>ⓘ</span>Model Information</div>
          <div className="nav-item"><span>⚙</span>Settings</div>
        </nav>
        <div className="sidebar-art"><div className="lung-symbol">◖ ◗</div><div className="pulse-line"/></div>
        <div className="sidebar-note"><b>LungSight AI</b><p>AI-powered lung image analysis for educational research.</p><small>RESEARCH PROTOTYPE</small></div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="search-box"><span>⌕</span><input placeholder="Search patients, reports, or analyses..." aria-label="Search"/></div>
          <div className="topbar-right"><span className="online"><i/>System Online</span><span className="bell">♧</span><span className="avatar">R</span><span className="user-label"><b>Radiology Workspace</b><small>Research user</small></span><span className="moon">◐</span></div>
        </header>

        <section className="hero">
          <div className="hero-copy"><small>AI-ASSISTED LUNG IMAGING</small><h1>AI for<br/>Healthier Lungs</h1><p>Early exploration. Model-assisted analysis. Better understanding.</p></div>
          <div className="hero-art" aria-hidden="true"><div className="orbit">✧</div><div className="lungs">◖　◗</div></div>
          <div className="hero-quote"><b>“Smarter AI.<br/>Clearer insights.<br/>Responsible research.”</b><i/><span>✧ &nbsp; AI-assisted analysis</span><span>⬡ &nbsp; Visual explanations</span><span>♡ &nbsp; Research insights</span></div>
          <div className="hero-stats">
            <div className="stat"><em>♟</em><div><b>{result ? '01' : '—'}</b><small>Scans analyzed</small></div></div>
            <div className="stat"><em>∿</em><div><b>{result ? 'Ready' : '—'}</b><small>AI analysis</small></div></div>
            <div className="stat"><em>◷</em><div><b>{busy ? 'Running' : 'Ready'}</b><small>Pipeline status</small></div></div>
            <div className="stat"><em>⬡</em><div><b>Research</b><small>Model validation</small></div></div>
          </div>
        </section>

        <section className="steps">
          {['Upload image','Preprocessing','AI analysis','Visual explanation','Results & report'].map((s,i)=><div className={`step ${i===0&&!result?'step-current':''} ${result&&i<3?'step-complete':''}`} key={s}><span>{i+1}</span><div><b>{s}</b><small>{['Choose chest X-ray','Resize and normalize','Class prediction','Grad-CAM heatmap','Review model output'][i]}</small></div>{i<4&&<i>›</i>}</div>)}
        </section>

        <section className="workspace">
          <article className="panel upload-panel">
            <div className="panel-heading"><h2>▣ &nbsp;Upload Chest X-ray</h2><span className="pill">JPG / PNG / WEBP</span></div>
            <div className={`dropzone ${preview?'loaded':''}`} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();handleFile(e.dataTransfer.files?.[0]);}}>
              <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp" onChange={e=>handleFile(e.target.files?.[0])}/>
              {preview?<><img src={preview} alt="Selected chest X-ray"/><b>{file?.name}</b><small>Image selected and ready</small></>:<><div className="upload-icon">⇧</div><b>Drag & drop a chest X-ray image</b><span>or click below to browse files</span><small>Supports JPG, PNG, WebP · max 15 MB</small></>}
              <button className="primary choose" onClick={chooseFile}>{preview?'Choose another file':'Choose File'}</button>
            </div>
            <div className="button-row"><button className="secondary" onClick={clearImage} disabled={!file||busy}>Clear image</button><button className="primary" onClick={analyzeImage} disabled={!file||busy}>{busy?'Analyzing…':'▶ Run Analysis'}</button></div>
            {error&&<div className="error" role="alert">{error}</div>}
          </article>

          <article className="panel preview-panel">
            <div className="panel-heading"><h2>▧ &nbsp;Image Preview</h2>{preview&&<button className="close" onClick={clearImage}>×</button>}</div>
            <div className="preview-box">{preview?<img src={preview} alt="Image preview"/>:<div><span>◉</span><b>Image preview</b><small>Your selected scan appears here</small></div>}</div>
            <div className="file-meta"><i className={file?'dot-green':'dot-muted'}/>{file?file.name:'No image selected'}<small>{file?`${(file.size/1024).toFixed(0)} KB`:'Waiting'}</small></div>
          </article>

          <article className="panel patient-panel">
            <div className="panel-heading"><h2>♟ &nbsp;Patient Information</h2></div>
            <label>Patient ID<input value={patientId} onChange={e=>setPatientId(e.target.value)}/></label>
            <div className="two-fields"><label>Age<input value={age} onChange={e=>setAge(e.target.value)}/></label><label>Gender<select value={gender} onChange={e=>setGender(e.target.value)}><option>Other</option><option>Male</option><option>Female</option></select></label></div>
            <label>Modality<select defaultValue="Chest X-ray (CXR)"><option>Chest X-ray (CXR)</option><option>Other image</option></select></label>
            <label>View Position<select defaultValue="Not specified"><option>Not specified</option><option>PA View</option><option>AP View</option></select></label>
            <small className="hint">Demo fields only; not sent to the model API.</small>
          </article>

          <article className="panel pipeline-panel">
            <div className="panel-heading"><h2>⌁ &nbsp;Analysis Pipeline</h2></div>
            <div className="pipeline">{['Image validation','Preprocessing','AI inference','Grad-CAM explanation','Generate report'].map((s,i)=><div className="pipeline-row" key={s}><span className={result&&i<3?'done':i===0?'now':''}>{result&&i<3?'✓':i+1}</span><div><b>{s}</b><small>{['Checking file format and size','Resize and normalize image','Model class prediction','Visualize model attention','Download analysis summary'][i]}</small></div></div>)}</div>
            <button className="primary run-wide" onClick={analyzeImage} disabled={!file||busy}>{busy?'Processing…':'▶ Run Analysis'}</button>
          </article>
        </section>

        <section className="lower-grid">
          <article className="panel analysis-panel">
            <div className="panel-heading"><h2>▦ &nbsp;Analysis Results</h2><span className="pill">{result?'RESPONSE RECEIVED':'AWAITING ANALYSIS'}</span></div>
            <div className="tabs">{views.map(v=><button key={v.label} className={activeView===v.label?'selected-tab':''} disabled={!v.src} onClick={()=>setActiveView(v.label)}>{v.label}</button>)}</div>
            <div className="analysis-image">{currentImage?<img src={currentImage} alt={activeView}/>:<div><span>▧</span><b>{result?'Visualization unavailable':'Run analysis to view results'}</b><small>Original image and Grad-CAM outputs appear here.</small></div>}</div>
            <div className="image-caption">{activeView}{activeView!=='Original image'&&<small>Attention visualization only—not a verified tumour boundary or segmentation mask.</small>}</div>
          </article>

          <article className="panel prediction-panel">
            <div className="panel-heading"><h2>⬡ &nbsp;Prediction Results</h2></div>
            {result?<><div className="prediction-name"><i className="dot-green"/><b>{result.predicted_class}</b><span>{risk} label</span></div><div className="scores">{(scores.length?scores:[['Model confidence',confidence]]).map(([name,value])=><div className="score" key={name}><span>{name}</span><div><i style={{width:`${pct(value)}%`}}/></div><b>{pct(value).toFixed(1)}%</b></div>)}</div><div className="metrics"><div><small>Predicted class</small><b>{result.predicted_class}</b></div><div><small>Model confidence</small><b>{confidence.toFixed(1)}%</b></div></div><p className="disclaimer">{result.risk_score_note||'Model scores are not clinically validated probabilities or personal cancer-risk estimates.'}</p></>:<div className="empty"><span>◉</span><b>Prediction will appear here</b><small>Run the model to view class scores.</small></div>}
          </article>

          <article className="panel metrics-panel">
            <div className="panel-heading"><h2>♧ &nbsp;Lung Metrics</h2></div>
            <div className="metric"><span>◉</span><div><b>Malignant class score</b><small>Model output only</small></div><strong>{result?`${malignant.toFixed(1)}%`:'—'}</strong></div>
            <div className="metric"><span>⬡</span><div><b>Risk category</b><small>Demo threshold label</small></div><strong>{result?risk:'—'}</strong></div>
            <div className="metric"><span>▧</span><div><b>Input image</b><small>Uploaded file</small></div><strong>{file?'Ready':'—'}</strong></div>
            <div className="actions-title">ACTIONS</div>
            <button className="outline-action" onClick={downloadReport} disabled={!result||reportBusy}>⇩ &nbsp;{reportBusy?'Preparing PDF…':'Download PDF Report'}</button>
            <button className="outline-action" onClick={shareResult} disabled={!result}>↗ &nbsp;Share Result</button>
          </article>
        </section>
        <footer className="footer"><span>LungSight AI · Student research project</span><span>Educational use only · Not for clinical decisions</span></footer>
      </main>
    </div>
  );
}
