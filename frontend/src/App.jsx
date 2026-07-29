import React, { useState, useEffect, useRef } from "react";
import TokenSummaryPanel from "./components/TokenSummaryPanel";
import CharacterSidebar from "./components/CharacterSidebar";
import ShotCard from "./components/ShotCard";

const API_BASE = window.location.hostname === "127.0.0.1" ? "http://127.0.0.1:8000" : "http://localhost:8000";

const SAMPLE_SCRIPT = `SCENE 1 — EXT. VILLAGE TEA STALL — MORNING

A colorful tea stall in a dusty Indian village. JACKIE SHROFF strolls through the lane in his signature style — sunglasses, white kurta, sleeves rolled up.

A young girl, MEERA (8), sits on a bench looking disappointed at a broken biscuit.

JACKIE
(noticing her)
Arre bachchi, udaas kyun? Ye le — Parle-G!

Meera grabs the pack, breaks a biscuit, and dunks it in her chai. Her face lights up.


SCENE 2 — EXT. VILLAGE SCHOOL PLAYGROUND — AFTERNOON

The school playground is alive with children. MEERA runs in, waving the pack of Parle-G, sharing biscuits with her friends.

JACKIE watches from a distance, leaning against a neem tree, smiling.

JACKIE (V.O.)
G maane Genius. Aur Genius ka matlab — Parle-G.`;

export default function App() {
  const [scriptText, setScriptText] = useState(SAMPLE_SCRIPT);
  const [scriptId, setScriptId] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [characters, setCharacters] = useState([]);

  // Model selection state
  const [selectedModel, setSelectedModel] = useState("gemini-3.5-flash");

  // Generation & polling state
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [progress, setProgress] = useState(0);
  const [tokenSummary, setTokenSummary] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const pollRef = useRef(null);

  // Poll status endpoint every 1.5 seconds while generation is in progress
  useEffect(() => {
    if (scriptId && isProcessing) {
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/storyboard/${scriptId}/status`);
          if (res.ok) {
            const data = await res.json();
            setProgress(data.progress || 0);
            if (data.token_summary) {
              setTokenSummary(data.token_summary);
            }

            // Update shot images, status, and error from status endpoint
            if (data.shots && data.shots.length > 0) {
              setScenes((prevScenes) =>
                prevScenes.map((scene) => ({
                  ...scene,
                  shots: scene.shots.map((shot) => {
                    const statusShot = data.shots.find((s) => s.shot_id === shot.id);
                    if (statusShot) {
                      return {
                        ...shot,
                        image_path: statusShot.image_path,
                        status: statusShot.status,
                        error: statusShot.error,
                      };
                    }
                    return shot;
                  }),
                }))
              );
            }

            if (data.status === "completed") {
              setIsProcessing(false);
              setStatusMessage("Storyboard Generation Complete!");
              clearInterval(pollRef.current);
            } else if (data.status === "failed") {
              setIsProcessing(false);
              setStatusMessage("Storyboard Generation Finished with some failures.");
              clearInterval(pollRef.current);
            }
          }
        } catch (err) {
          console.error("Status poll error:", err);
        }
      }, 1500);
    }

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [scriptId, isProcessing]);

  const handleParseAndGenerate = async () => {
    if (!scriptText.trim()) return;

    setIsProcessing(true);
    setErrorMessage(null);
    setStatusMessage("Parsing screenplay locally...");
    setProgress(0.05);

    try {
      // 1. POST /api/script/parse
      const parseRes = await fetch(`${API_BASE}/api/script/parse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          script_text: scriptText.trim(),
          model: selectedModel
        }),
      });

      if (!parseRes.ok) {
        const errData = await parseRes.json();
        throw new Error(errData.detail || "Failed to parse script");
      }

      const parseData = await parseRes.json();
      const newScriptId = parseData.script_id;
      setScriptId(newScriptId);
      setScenes(parseData.scenes || []);
      setCharacters(parseData.characters || []);

      setStatusMessage("Starting storyboard generation...");
      setProgress(0.15);

      // 2. POST /api/storyboard/generate/{newScriptId}
      const genRes = await fetch(`${API_BASE}/api/storyboard/generate/${newScriptId}`, {
        method: "POST",
      });

      if (!genRes.ok) {
        const errData = await genRes.json();
        throw new Error(errData.detail || "Failed to start storyboard generation");
      }

      setStatusMessage("Generating storyboard images...");
      setProgress(0.2);
      
    } catch (err) {
      setErrorMessage(err.message);
      setIsProcessing(false);
    }
  };

  const handleShotEditSuccess = (shotId, newImagePath) => {
    // Update local state image path for edited shot
    setScenes((prevScenes) =>
      prevScenes.map((scene) => ({
        ...scene,
        shots: scene.shots.map((shot) =>
          shot.id === shotId ? { ...shot, image_path: newImagePath, status: "completed", error: null } : shot
        ),
      }))
    );

    // Refresh token summary from backend status
    if (scriptId) {
      fetch(`${API_BASE}/api/storyboard/${scriptId}/status`)
        .then((res) => res.json())
        .then((data) => {
          if (data.token_summary) setTokenSummary(data.token_summary);
        })
        .catch(console.error);
    }
  };

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setScriptId(null);
    setScenes([]);
    setCharacters([]);
    setIsProcessing(false);
    setStatusMessage("");
    setProgress(0);
    setTokenSummary(null);
    setErrorMessage(null);
  };

  return (
    <div className="app-container">

      <header className="app-header">
        <div>
          <h1 className="app-title">AI Storyboard Generator</h1>
          <p className="app-subtitle">
            Anchor-Based Character Consistency & Prompt Cache Token Optimization
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <div className="model-selector-container">
            <label htmlFor="model-select" style={{ fontSize: "0.85rem", fontWeight: "600", marginRight: "0.5rem", color: "var(--text-sub)" }}>
              Model:
            </label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="model-select"
              disabled={isProcessing || scriptId !== null}
            >
              <optgroup label="Gemini">
                <option value="gemini-3.6-flash">Gemini-3.6-flash</option>
                <option value="gemini-3.5-flash">Gemini-3.5-flash</option>
                <option value="gemini-2.5-flash">Gemini-2.5-flash</option>
                <option value="gemini-3.1-pro">Gemini-3.1-pro</option>
              </optgroup>
              <optgroup label="OpenAI">
                <option value="gpt-image-1">GPT Image 1</option>
                <option value="gpt-image-1.5">GPT Image 1.5</option>
              </optgroup>
            </select>
          </div>
          {scriptId && (
            <button className="btn btn-secondary btn-sm" onClick={handleReset}>
              ↺ New Script
            </button>
          )}
        </div>
      </header>


      <main className="main-content">
        <div className="content-left">

          <TokenSummaryPanel tokenSummary={tokenSummary} />

          {errorMessage && <div className="error-box">{errorMessage}</div>}


          {!scriptId && (
            <div className="card" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <h2 style={{ fontSize: "1.2rem", fontWeight: "600", color: "#a5b4fc" }}>
                1. Input Screenplay / Script
              </h2>
              <p style={{ fontSize: "0.875rem", color: "#94a3b8" }}>
                Paste a screenplay with 2-4 scenes. The system will automatically parse the screenplay, extract scenes, shots, characters, and generate the storyboard images.
              </p>
              <textarea
                className="script-textarea"
                value={scriptText}
                onChange={(e) => setScriptText(e.target.value)}
                placeholder="Paste screenplay here..."
                disabled={isProcessing}
              />
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button
                  className="btn"
                  onClick={handleParseAndGenerate}
                  disabled={isProcessing || !scriptText.trim()}
                >
                  {isProcessing ? "Processing..." : "Generate Storyboard 🚀"}
                </button>
              </div>
            </div>
          )}


          {isProcessing && (
            <div className="card progress-container">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontWeight: "600", fontSize: "0.95rem" }}>{statusMessage}</span>
                <span style={{ fontSize: "0.85rem", color: "#10b981", fontWeight: "700" }}>
                  {Math.round(progress * 100)}%
                </span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${Math.max(progress * 100, 5)}%` }} />
              </div>
            </div>
          )}


          {scenes.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h2 style={{ fontSize: "1.25rem", fontWeight: "700", color: "#f8fafc" }}>
                  Generated Storyboard Shots
                </h2>
              </div>

              {scenes.map((scene) => (
                <div key={scene.id} className="scene-section">
                  <div className="scene-header">
                    <span className="scene-title">
                      SCENE {scene.scene_number}: {scene.location}
                    </span>
                    <span className="scene-meta">
                      {scene.time_of_day} | {scene.weather || "clear"} | mood: {scene.mood || "neutral"}
                    </span>
                  </div>

                  <div className="shots-grid">
                    {scene.shots && scene.shots.map((shot) => (
                      <ShotCard
                        key={shot.id}
                        shot={shot}
                        sceneNumber={scene.scene_number}
                        onShotEditSuccess={handleShotEditSuccess}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>


        <div className="content-right">
          <CharacterSidebar characters={characters} />
        </div>
      </main>
    </div>
  );
}
