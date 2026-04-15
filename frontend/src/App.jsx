import { useState, useEffect } from 'react'
import './App.css'
import Loading from './components/Loading'

const API_BASE_URL = 'http://127.0.0.1:8000'

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme')
    return savedTheme || 'light'
  })

  useEffect(() => {
    const root = document.documentElement
    root.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    return () => {
      if (result?.url) {
        URL.revokeObjectURL(result.url)
      }
    }
  }, [result])

  async function handleClick(e) {
    e.preventDefault()
    setIsLoading(true)
    // extract the contents of the form
    const form = e.currentTarget
    const modelFile = form.elements.model?.files?.[0] ?? null
    const frameFile = form.elements.frame?.files?.[0] ?? null
    const inferenceMode = form.elements.inference_mode?.value ?? ''
    const densityThresholdRaw = form.elements.density_threshold?.value ?? ''
    const densityThreshold = Number(densityThresholdRaw)

    if (modelFile == null || frameFile == null || inferenceMode == null) {
      setIsLoading(false)
      alert('Seems that your files were `null` at one point. Please, review your input')
      return
    }

    if (!inferenceMode) {
      setIsLoading(false)
      alert('Please select an inference mode')
      return
    }

    if (!Number.isFinite(densityThreshold) || densityThreshold < 0 || densityThreshold > 100) {
      setIsLoading(false)
      alert('Please enter a valid density threshold between 0 and 100')
      return
    }

    try {
      setResult(null)

      const payload = new FormData()
      payload.append('model', modelFile)
      payload.append('frame', frameFile)
      payload.append('inference_mode', inferenceMode)
      payload.append('density_threshold', densityThreshold.toString())

      const response = await fetch(`${API_BASE_URL}/inference`, {
        method: 'POST',
        body: payload,
      })

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`)
      }

      const data = await response.json()
      const apiResults = data?.results
      if (!apiResults || !Array.isArray(apiResults.logs)) {
        throw new Error('Response does not include logs')
      }

      setResult(apiResults)
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Unexpected error while processing request')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main>
      <div className="theme-toggle">
        <button 
          className={`theme-btn light-btn ${theme === 'light' ? 'active' : ''}`}
          onClick={() => setTheme('light')}
          title="Modo claro"
        >
          ☀
        </button>
        <button 
          className={`theme-btn dark-btn ${theme === 'dark' ? 'active' : ''}`}
          onClick={() => setTheme('dark')}
          title="Modo oscuro"
        >
          ☾
        </button>
      </div>
      <header className="app-header">
        <img src="/uc3m-logo.jpg" alt="UC3M Logo" className="uc3m-logo" />
        <h1>CARLA User Interface</h1>
        <p className="app-description">
          This UI was done for easing the video processing for box prediction and geo-spatial location for the Final Bachelor Thesis of Javier Martín Pizarro.
          <br />
          <br />
          The usage is simple: insert your .pt model and select if either it was trained or not using tiling methodologies. Then, insert your data to be predicted: it can be either an image or a video (mp4 file).
        </p>
      </header>

      <div className="content">
        <section className="form-section">
          <form onSubmit={handleClick} className="inference-form">
            <fieldset>
              <legend>Model Configuration</legend>
              
              <div className="form-group">
                <label htmlFor="model">Model file (.pt)</label>
                <input type="file" id="model" name="model" accept=".pt" />
              </div>

              <div className="form-group">
                <span className="radio-group-label">Inference mode</span>
                <div className="radio-group">
                  <label className="radio-option">
                    <input type="radio" id="tiled" name="inference_mode" value="Tiled" />
                    <span>Tiled</span>
                  </label>
                  <label className="radio-option">
                    <input type="radio" id="not_tiled" name="inference_mode" value="NonTiled" />
                    <span>Non-Tiled</span>
                  </label>
                </div>
              </div>
            </fieldset>

            <fieldset>
              <legend>Input Data</legend>
              <div className="form-group">
                <label htmlFor="frame">Image or Video file</label>
                <input type="file" id="frame" name="frame" accept="image/*,video/*" />
              </div>

              <div className="form-group">
                <label htmlFor="density_threshold">Density threshold (% ocupacion)</label>
                <input
                  type="number"
                  id="density_threshold"
                  name="density_threshold"
                  min="0"
                  max="100"
                  step="0.1"
                  defaultValue="10"
                  required
                />
              </div>
            </fieldset>

            <button type="submit" className="submit-btn">Process file</button>
            {isLoading && (
              <div className="loading">
                <Loading color="#111" size={56} />
              </div>
            )}
          </form>
        </section> 

        {result && (
          <section className="results-section">
            <h2>Logs de densidad</h2>
            <p>
              Umbral: <strong>{result.density_threshold}%</strong> de ocupacion. Ventana: <strong>{result.window_seconds}s</strong> ({result.window_frames} frames).
            </p>

            {result.saved_artifacts && (
              <div className="artifacts-block">
                <h3>Videos generados</h3>
                <ul className="artifacts-list">
                  <li>
                    <strong>Predicciones:</strong> {result.saved_artifacts.output_media_path}
                  </li>
                  <li>
                    <strong>Plots:</strong> {result.saved_artifacts.simple_plots_media_path}
                  </li>
                  <li>
                    <strong>Carpeta:</strong> {result.saved_artifacts.run_dir}
                  </li>
                </ul>
              </div>
            )}

            {result.logs.length > 0 ? (
              <ul className="logs-list">
                {result.logs.map((logItem, index) => (
                  <li key={`${index}-${logItem}`}>{logItem}</li>
                ))}
              </ul>
            ) : (
              <p>No se detectaron ventanas de 3s por encima del umbral.</p>
            )}
          </section>
        )}
      </div>
    </main>
  )
}

export default App
