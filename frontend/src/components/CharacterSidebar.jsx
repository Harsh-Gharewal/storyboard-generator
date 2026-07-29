import React from "react";

const API_BASE = window.location.hostname === "127.0.0.1" ? "http://127.0.0.1:8000" : "http://localhost:8000";

export default function CharacterSidebar({ characters }) {
  if (!characters || characters.length === 0) {
    return (
      <div className="card">
        <h3 className="scene-title" style={{ marginBottom: "0.5rem" }}>Character Bible</h3>
        <p className="character-desc">No characters extracted yet.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="scene-title" style={{ marginBottom: "1rem" }}>
        Character Bible ({characters.length})
      </h3>
      <div className="character-list">
        {characters.map((char) => {
          const imgUrl = char.anchor_image_path
            ? `${API_BASE}/${char.anchor_image_path}`
            : null;

          return (
            <div key={char.id} className="character-card">
              {imgUrl ? (
                <img
                  src={imgUrl}
                  alt={char.name}
                  className="character-anchor-img"
                  onError={(e) => {
                    e.target.style.display = "none";
                  }}
                />
              ) : (
                <div className="character-anchor-img" style={{ display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.7rem", color: "#64748b", textAlign: "center" }}>
                  Anchor Gen...
                </div>
              )}
              <div className="character-info">
                <span className="character-name">{char.name}</span>
                <p className="character-desc">{char.description}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
