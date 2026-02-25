import { useEffect, useState } from 'react'
import './App.css'
import Loading from './components/Loading'

const API_BASE_URL = 'http://127.0.0.1:8000'

function b64ToBlob(base64Content, contentType) {
  const binary = atob(base64Content)
  const bytes = new Uint8Array(binary.length)

  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }

  return new Blob([bytes], { type: contentType })
}

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [simplePlotsResult, setSimplePlotsResult] = useState(null)
  const [outputResult, setOutputResult] = useState(null)

  useEffect(() => {
    return () => {
      if (simplePlotsResult?.url) {
        URL.revokeObjectURL(simplePlotsResult.url)
      }
      if (outputResult?.url) {
        URL.revokeObjectURL(outputResult.url)
      }
    }
  }, [simplePlotsResult, outputResult])

  async function handleClick(e) {
    e.preventDefault()
    setIsLoading(true)
    // extract the contents of the form
    const form = e.currentTarget
    const modelFile = form.elements.model?.files?.[0] ?? null
    const frameFile = form.elements.frame?.files?.[0] ?? null
    const inferenceMode = form.elements.inference_mode?.value ?? ''

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

    try {
      if (simplePlotsResult?.url) {
        URL.revokeObjectURL(simplePlotsResult.url)
      }
      if (outputResult?.url) {
        URL.revokeObjectURL(outputResult.url)
      }
      setSimplePlotsResult(null)
      setOutputResult(null)

      const payload = new FormData()
      payload.append('model', modelFile)
      payload.append('frame', frameFile)
      payload.append('inference_mode', inferenceMode)

      const response = await fetch(`${API_BASE_URL}/inference`, {
        method: 'POST',
        body: payload,
      })

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`)
      }

      const data = await response.json()
      const simplePlots = data?.results?.simple_plots
      const outputVideo = data?.results?.output_video

      if (
        !simplePlots?.content_base64
        || !outputVideo?.content_base64
      ) {
        throw new Error('Response does not include plot results')
      }

      const simplePlotsBlob = b64ToBlob(
        simplePlots.content_base64,
        simplePlots.content_type ?? 'application/octet-stream',
      )
      const outputVideoBlob = b64ToBlob(
        outputVideo.content_base64,
        outputVideo.content_type ?? 'application/octet-stream',
      )

      setSimplePlotsResult({
        url: URL.createObjectURL(simplePlotsBlob),
        contentType: simplePlots.content_type ?? '',
        title: simplePlots.filename ?? 'simple_plots',
      })
      setOutputResult({
        url: URL.createObjectURL(outputVideoBlob),
        contentType: outputVideo.content_type ?? '',
        title: outputVideo.filename ?? 'output',
      })
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Unexpected error while processing request')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main>
      <h1>CARLA User Interface</h1>
      <div>
        <section>
          <form onSubmit={handleClick}>
            <div>
              <input type="file" id='model' name="model" accept=".pt" />
              <br />
              <label htmlFor="model">Model in .pt format</label><br />
              
              <input type="radio" id="tiled" name="inference_mode" value="Tiled" />
              <label htmlFor="tiled">Model uses tiling</label><br />
              <input type="radio" id="not_tiled" name="inference_mode" value="NonTiled" />
              <label htmlFor="not_tiled">Model does not use tiling</label><br/>

            </div>

            <div>
              <input type="file" id='frame' name="frame" accept="image/*,video/*" />
              <br />
              <label htmlFor="frame">File, it can be either an image or a video</label>
            </div>

            <button type='submit'>Process file</button>
            {isLoading && (
              <div className="loading">
                <Loading color="#111" size={56} />
              </div>
            )}
          </form>
        </section> 

        {simplePlotsResult && outputResult && (
          <section>
            <h2>Plots</h2>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <div>
                <h3>{simplePlotsResult.title}</h3>
                {simplePlotsResult.contentType.startsWith('image/') ? (
                  <img src={simplePlotsResult.url} alt={simplePlotsResult.title} width="480" />
                ) : (
                  <video src={simplePlotsResult.url} controls autoPlay muted playsInline width="480" />
                )}
              </div>
              <div>
                <h3>{outputResult.title}</h3>
                {outputResult.contentType.startsWith('image/') ? (
                  <img src={outputResult.url} alt={outputResult.title} width="480" />
                ) : (
                  <video src={outputResult.url} controls autoPlay muted playsInline width="480" />
                )}
              </div>
            </div>
          </section>
        )}
      </div>
    </main>
  )
}

export default App
