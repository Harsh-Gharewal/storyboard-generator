import React, { useState } from "react";

const API_BASE = window.location.hostname === "127.0.0.1" ? "http://127.0.0.1:8000" : "http://localhost:8000";

export default function ShotCard({ shot, sceneNumber, onShotEditSuccess }) {
  const [instruction, setInstruction] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [editError, setEditError] = useState(null);

  // Version history drawer state
  const [showVersions, setShowVersions] = useState(false);
  const [versions, setVersions] = useState([]);
  const [loadingVersions, setLoadingVersions] = useState(false);

  // Generate state
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState(null);

  const currentImgUrl = shot.image_path ? `${API_BASE}/${shot.image_path}` : null;

  const handleGenerate = async () => {
    setIsGenerating(true);
    setGenerateError(null);
    try {
      const res = await fetch(`${API_BASE}/api/shot/${shot.id}/generate`, {
        method: "POST"
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to generate shot image");
      }
      const data = await res.json();
      if (data.status === "done" && data.image_path) {
        if (onShotEditSuccess) {
          onShotEditSuccess(shot.id, data.image_path);
        }
      } else {
        throw new Error(data.error || "Generation failed");
      }
    } catch (err) {
      setGenerateError(err.message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleEditSubmit = async (e) => {
    e.preventDefault();
    if (!instruction.trim()) return;

    setIsEditing(true);
    setEditError(null);

    try {
      const response = await fetch(`${API_BASE}/api/shot/${shot.id}/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instruction: instruction.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to edit shot image");
      }

      const data = await response.json();
      setInstruction("");
      if (onShotEditSuccess) {
        onShotEditSuccess(shot.id, data.new_version.image_path);
      }
      // If version drawer is open, refresh version list
      if (showVersions) {
        fetchVersions();
      }
    } catch (err) {
      setEditError(err.message);
    } finally {
      setIsEditing(false);
    }
  };

  const fetchVersions = async () => {
    setLoadingVersions(true);
    try {
      const response = await fetch(`${API_BASE}/api/shot/${shot.id}/versions`);
      if (response.ok) {
        const data = await response.json();
        setVersions(data.versions || []);
      }
    } catch (err) {
      console.error("Failed to load versions", err);
    } finally {
      setLoadingVersions(false);
    }
  };

  const toggleVersions = () => {
    if (!showVersions) {
      fetchVersions();
    }
    setShowVersions(!showVersions);
  };

  return (
    <div className="shot-card">
      <div className="shot-image-container">
        {currentImgUrl ? (
          <img
            src={`${currentImgUrl}?t=${Date.now()}`}
            alt={`Shot ${shot.shot_number}`}
            className="shot-image"
          />
        ) : shot.status === "failed" ? (
          <div className="shot-image-placeholder failed" style={{ borderColor: "#ef4444", backgroundColor: "#311818", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
            <span style={{ color: "#f87171", fontWeight: "600", padding: "1rem", textAlign: "center" }}>
              ⚠️ Generation Failed
              {shot.error && <div style={{ fontSize: "0.75rem", marginTop: "0.5rem", fontWeight: "normal", color: "#fca5a5" }}>{shot.error}</div>}
            </span>
          </div>
        ) : (
          <div className="shot-image-placeholder" style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
            {isGenerating ? (
              <span>Generating...</span>
            ) : (
              <button className="btn btn-sm" onClick={handleGenerate} style={{ backgroundColor: "#3b82f6", color: "white" }}>
                Generate Shot 📸
              </button>
            )}
          </div>
        )}
        <div className="shot-badge">
          Scene {sceneNumber} — Shot {shot.shot_number}
        </div>
      </div>

      <div className="shot-body">
        <div className="shot-meta-row">
          {shot.camera_angle && (
            <span className="tag">📷 {shot.camera_angle}</span>
          )}
          {shot.character_names && shot.character_names.map((name, idx) => (
            <span key={idx} className="tag tag-character">
              👤 {name}
            </span>
          ))}
        </div>

        <p className="shot-desc">{shot.description}</p>

        {editError && <div className="error-box" style={{ fontSize: "0.8rem", padding: "0.4rem" }}>{editError}</div>}

        {shot.status === "failed" && (
          <div className="retry-container" style={{ margin: "0.5rem 0", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {generateError && (
              <div className="error-box" style={{ fontSize: "0.8rem", padding: "0.4rem" }}>
                {generateError}
              </div>
            )}
            <button
              type="button"
              className="btn btn-sm"
              style={{ backgroundColor: "#ef4444", color: "#ffffff", width: "100%", fontWeight: "600" }}
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              {isGenerating ? "Retrying..." : "Retry Shot Generation ↺"}
            </button>
          </div>
        )}

        <form onSubmit={handleEditSubmit} className="edit-form">
          <input
            type="text"
            className="edit-input"
            placeholder="Edit (e.g. 'make it rainy', 'change camera angle')..."
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            disabled={isEditing || !shot.image_path}
          />
          <button
            type="submit"
            className="btn btn-sm"
            disabled={isEditing || !instruction.trim() || !shot.image_path}
          >
            {isEditing ? "Editing..." : "Apply Edit"}
          </button>
        </form>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "0.5rem" }}>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={toggleVersions}
          >
            {showVersions ? "Hide History" : "Version History"}
          </button>
        </div>
      </div>

      {showVersions && (
        <div className="version-drawer">
          <span style={{ fontSize: "0.8rem", fontWeight: "600", color: "#a5b4fc" }}>
            Filmstrip Version History ({versions.length})
          </span>
          {loadingVersions ? (
            <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Loading history...</div>
          ) : (
            <div className="version-filmstrip">
              {versions.map((ver) => (
                <div key={ver.id} className="version-thumb-card">
                  <img
                    src={`${API_BASE}/${ver.image_path}`}
                    alt={`v${ver.version_number}`}
                    className="version-thumb-img"
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "600" }}>
                    <span>v{ver.version_number}</span>
                    <span style={{ color: "#10b981" }}>{ver.token_usage?.requested || 0} tks</span>
                  </div>
                  {ver.edit_instruction && (
                    <div style={{ fontSize: "0.7rem", color: "#f59e0b", fontStyle: "italic" }}>
                      "{ver.edit_instruction}"
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
