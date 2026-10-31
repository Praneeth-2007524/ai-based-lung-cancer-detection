import { useRef, useState } from 'react';
import './styles.css';

const API_URL = 'https://ai-based-lung-cancer-detection-1.onrender.com';
const MAX_FILE_SIZE = 15 * 1024 * 1024;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

function toPercentage(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const percentage = numeric >= 0 && numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(100, percentage));
}

export default function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  function chooseFile() {
    inputRef.current?.click();
  }

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

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
  }

  function clearImage() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl('');
    setResult(null);
    setError('');
    if (inputRef.current) inputRef.current.value = '';
  }

  async function analyzeImage() {
    if (!file) {
      setError('Choose an image before starting the analysis.');
      return;
    }

    setBusy(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const responseText = await response.text();
      let data;

      try {
        data = JSON.parse(responseText);
      } catch {
        throw new Error(`The server returned an invalid response (HTTP ${response.status}).`);
      }

      if (!response.ok) {
        throw new Error(
          data.detail || data.message || `Server error: HTTP ${response.status}`
        );
      }

      const predictedClass =
        data.predicted_class ||
        data.prediction ||
        data.class_name ||
        data.predicted_label;

      if (!predictedClass) {
        throw new Error('The API response did not contain a prediction.');
      }

      setResult({
        ...data,
        predicted_class: predictedClass,
      });
    } catch (err) {
      setError(
        `${err.message || 'Could not connect to the model API.'} The free Render service may need time to wake up.`
      );
    } finally {
      setBusy(false);
    }
  }

  const confidenceValue = result?.confidence_percent ?? result?.confidence;
  const rawConfidence = Number(confidenceValue);
  const safeConfidence = Number.isFinite(rawConfidence)
    ? Math.max(0, Math.min(100, result?.confidence_percent != null ? rawConfidence : toPercentage(rawConfidence)))
    : 0;
  const scores = result?.scores && typeof result.scores === 'object'
    ? Object.entries(result.scores)
    : [];

  function downloadReport() {
    if (!result) return;
    const report = [
      'LungSight AI — Lung Image Analysis Report',
      `Date: ${new Date().toLocaleString()}`,
      `Image: ${file?.name || 'Uploaded scan'}`,
      `Predicted class: ${result.predicted_class}`,
      `Model confidence: ${safeConfidence.toFixed(2)}%`,
      ...(scores.length ? ['Class scores:', ...scores.map(([name, value]) => {
        const numeric = Number(value);
        const pct = toPercentage(numeric);
        return `- ${name}: ${pct.toFixed(2)}%`;
      })] : []),
      '',
      result.notice || 'Experimental model output. Not a medical diagnosis.',
      'For educational research only. Do not use this report for diagnosis or treatment decisions.'
    ].join('\n');
    const blob = new Blob([report], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'lungsight-analysis-report.txt';
    link.click();
    URL.revokeObjectURL(url);
  }

  async function shareReport() {
    if (!result) return;
    const summary = `LungSight AI analysis\nPrediction: ${result.predicted_class}\nConfidence: ${safeConfidence.toFixed(2)}%\nResearch use only — not a medical diagnosis.`;
    if (navigator.share) {
      try {
        await navigator.share({ title: 'LungSight AI Analysis', text: summary });
      } catch (err) {
        if (err?.name !== 'AbortError') setError('Unable to share this report on this device.');
      }
    } else if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(summary);
        setError('Report summary copied to clipboard.');
      } catch {
        setError('Sharing is not supported in this browser.');
      }
    } else {
      setError('Sharing is not supported in this browser.');
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">✚</div>
          <div>
            <div className="brand-name">LungSight <span>AI</span></div>
            <div className="brand-caption">LUNG ANALYSIS</div>
          </div>
        </div>

        <div className="nav-heading">WORKSPACE</div>
        <nav className="navigation" aria-label="Main navigation">
          <div className="nav-item active" aria-current="page">
            <span className="nav-icon">⌁</span>
            <span>Lung Analysis</span>
          </div>
        </nav>

        <div className="sidebar-note">
          <div className="note-icon">ⓘ</div>
          <strong>Research prototype</strong>
          <p>For learning and research only. This tool is not a medical device.</p>
        </div>
        <div className="sidebar-footer">AI · MEDICAL IMAGING</div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-title">
            <span className="topbar-mark">✚</span>
            <span>LungSight AI <small>/ Lung Analysis</small></span>
          </div>
          <div className="service-status"><span className="status-dot" /> Model API configured</div>
        </header>

        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow"><span /> AI-ASSISTED LUNG IMAGING</div>
            <h1>See deeper.<br /><span>Understand every scan.</span></h1>
            <p className="hero-copy">
              Upload a lung image and review the classification returned by your connected AI model.
            </p>
            <button type="button" className="button button-primary hero-button" onClick={chooseFile}>
              ＋ Start lung analysis
            </button>
          </div>
          <div className="hero-image-caption">
            <span className="caption-pulse" />
            LUNG IMAGING WORKSPACE
          </div>
        </section>

        <section className="stats-grid" aria-label="Analysis information">
          <article className="stat-card">
            <div className="stat-icon blue">▧</div>
            <div><div className="stat-label">SUPPORTED FORMATS</div><div className="stat-value">JPG · PNG</div><div className="stat-foot">WebP also accepted</div></div>
          </article>
          <article className="stat-card">
            <div className="stat-icon cyan">⌁</div>
            <div><div className="stat-label">MODEL API</div><div className="stat-value">Render</div><div className="stat-foot">Connected prediction service</div></div>
          </article>
          <article className="stat-card">
            <div className="stat-icon green">✓</div>
            <div><div className="stat-label">PROJECT STATUS</div><div className="stat-value">Research demo</div><div className="stat-foot">Not clinically validated</div></div>
          </article>
        </section>

        <section className="content-grid">
          <article className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <div className="section-kicker">01 / IMAGE INPUT</div>
                <h2>Upload lung scan</h2>
                <p>Select an image to preview it before analysis</p>
              </div>
              <span className="tag">MAX 15 MB</span>
            </div>

            <div
              className={`upload-zone ${previewUrl ? 'has-image' : ''}`}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                handleFile(event.dataTransfer.files?.[0]);
              }}
            >
              <input
                ref={inputRef}
                className="file-input"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(event) => handleFile(event.target.files?.[0])}
              />
              {previewUrl ? (
                <>
                  <img className="image-preview" src={previewUrl} alt="Selected lung scan preview" />
                  <div className="file-name">{file?.name}</div>
                  <div className="upload-description">Image selected and ready for analysis.</div>
                </>
              ) : (
                <>
                  <div className="upload-placeholder-art">
                    <img src="/images/lung-ct-scan.png" alt="" />
                    <span className="scan-corner scan-tl" /><span className="scan-corner scan-tr" />
                    <span className="scan-corner scan-bl" /><span className="scan-corner scan-br" />
                  </div>
                  <h3>Drop your lung scan here</h3>
                  <p>Choose a JPG, PNG, or WebP image from your computer.</p>
                </>
              )}
              <button type="button" className="button" onClick={chooseFile}>
                {previewUrl ? 'Choose another image' : 'Browse image files'}
              </button>
              <div className="upload-hint">Do not upload identifiable patient information.</div>
            </div>

            <div className="action-row">
              <button type="button" className="button button-quiet" onClick={clearImage} disabled={!file || busy}>
                Clear image
              </button>
              <button type="button" className="button button-primary" onClick={analyzeImage} disabled={!file || busy}>
                {busy ? 'Analyzing…' : 'Run AI analysis ↗'}
              </button>
            </div>
            {error && <div className="message error-message" role="alert">{error}</div>}
          </article>

          <article className="panel result-panel">
            <div className="panel-heading">
              <div>
                <div className="section-kicker">02 / AI OUTPUT</div>
                <h2>Analysis results</h2>
                <p>Classification returned by your model API</p>
              </div>
              <span className={`tag ${result ? 'tag-success' : ''}`}>{result ? 'Response received' : busy ? 'Processing' : 'Waiting'}</span>
            </div>

            {!result ? (
              <div className="empty-result">
                <div className="result-lung-art"><img src="/images/glowing-lungs.png" alt="" /></div>
                <h3>{busy ? 'Analyzing your scan' : 'Your results will appear here'}</h3>
                <p>{busy
                  ? 'Sending the image to the backend. The first request can take a little longer.'
                  : 'Upload a lung scan and run the model to view the predicted class and confidence score.'}</p>
                <div className="result-wait-line"><span /> Waiting for scan input</div>
              </div>
            ) : (
              <div className="result-content">
                <div className="result-status"><span>✓</span> ANALYSIS COMPLETE</div>
                <div className="result-kicker">MODEL CLASSIFICATION · RESEARCH ONLY</div>
                <div className="predicted-class">{result.predicted_class}</div>
                <p className="result-note">{result.notice || 'This is an experimental model output, not a medical diagnosis.'}</p>
                <div className="confidence-card">
                  <div className="confidence-line">
                    <span>Model confidence</span><strong>{safeConfidence.toFixed(2)}%</strong>
                  </div>
                  <div className="progress-track"><div className="progress-fill" style={{ width: `${safeConfidence}%` }} /></div>
                </div>
                {scores.length > 0 && (
                  <div className="score-list">
                    <div className="scores-heading">CLASS SCORES</div>
                    {scores.map(([name, value]) => {
                      const numeric = Number(value);
                      const percentage = toPercentage(numeric);
                      return (
                        <div className="score-item" key={name}>
                          <div className="confidence-line"><span>{name}</span><strong>{percentage.toFixed(2)}%</strong></div>
                          <div className="progress-track"><div className="progress-fill" style={{ width: `${percentage}%` }} /></div>
                        </div>
                      );
                    })}
                  </div>
                )}
                <div className="report-actions">
                  <button type="button" className="button button-primary" onClick={downloadReport}>↓ Download report</button>
                  <button type="button" className="button" onClick={shareReport}>↗ Share report</button>
                </div>
                <div className="medical-warning">⚠ Educational research only. Do not use this result for diagnosis or treatment decisions.</div>
              </div>
            )}
          </article>
        </section>

        <section className="bottom-grid">
          <article className="panel workflow-panel">
            <div className="panel-heading">
              <div><div className="section-kicker">HOW IT WORKS</div><h2>Analysis workflow</h2><p>From scan upload to model output</p></div>
              <span className="tag">3 STEPS</span>
            </div>
            <div className="workflow-steps">
              <div className="workflow-step"><span>01</span><div><strong>Upload</strong><p>Select a supported lung image.</p></div></div>
              <div className="workflow-step"><span>02</span><div><strong>Analyze</strong><p>The connected backend runs inference.</p></div></div>
              <div className="workflow-step"><span>03</span><div><strong>Review</strong><p>Review the returned class and scores.</p></div></div>
            </div>
          </article>
          <article className="panel deployment-panel">
            <div className="panel-heading">
              <div><div className="section-kicker">SYSTEM SETUP</div><h2>Deployment architecture</h2><p>Your existing frontend and backend services</p></div>
              <span className="tag">LIVE SETUP</span>
            </div>
            <div className="deployment-content">
              <div className="deployment-row"><span className="deployment-logo">V</span><div><strong>Vercel · Frontend</strong><p>React lung analysis interface</p></div><span className="service-label">UI</span></div>
              <div className="deployment-row"><span className="deployment-logo render-logo">R</span><div><strong>Render · Backend</strong><p>Python prediction API</p></div><span className="service-label">API</span></div>
              <p className="deployment-note">A successful response confirms the API returned a prediction; it does not establish clinical accuracy.</p>
            </div>
          </article>
        </section>

        <footer className="app-footer">
          <span>© LungSight AI · Student research project</span>
          <span>Research use only · Not for clinical decisions</span>
        </footer>
      </main>
    </div>
  );
}
