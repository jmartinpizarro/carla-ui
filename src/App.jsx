import { useState } from 'react'
import './App.css'
import Loading from './components/Loading'

function App() {
  const [isLoading, setIsLoading] = useState(false);

  function handleClick(e){
    e.preventDefault();
    setIsLoading(true);
    // extract the contents of the form
    const form = e.currentTarget;
    const modelFile = form.elements.model?.files?.[0] ?? null;
    const frameFile = form.elements.frame?.files?.[0] ?? null;
    const inferenceMode = form.elements.inference_mode?.value ?? '';

    if (modelFile == null || frameFile == null || inferenceMode == null) {
      setIsLoading(false);
      alert('Seems that your files were `null` at one point. Please, review your input')
      return;
    }
    
  } 

  return (
    <main>
      <h1>CARLA User Interface</h1>
      <div>
        <section>
          <form action="post" onSubmit={handleClick}>
            <div>
              <input type="file" id='model' name="model" />
              <br />
              <label htmlFor="model">Model in .pt format</label><br />
              
              <input type="radio" id="tiled" name="inference_mode" value="Tiled" />
              <label htmlFor="tiled">Model uses tiling</label><br />
              <input type="radio" id="not_tiled" name="inference_mode" value="NonTiled" />
              <label htmlFor="not_tiled">Model does not use tiling</label><br/>

            </div>

            <div>
              <input type="file" id='frame' name="frame" />
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
      </div>
    </main>
  )
}

export default App
