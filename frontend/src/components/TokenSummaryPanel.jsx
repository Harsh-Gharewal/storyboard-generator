import React from "react";

export default function TokenSummaryPanel({ tokenSummary }) {
  if (!tokenSummary) {
    return (
      <div className="token-banner">
        <div className="metric-item">
          <span className="metric-label">Total Prompt Tokens</span>
          <span className="metric-value">0</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Tokens Cached</span>
          <span className="metric-value">0</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Fresh Tokens</span>
          <span className="metric-value">0</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Savings %</span>
          <span className="metric-value highlight">0.0%</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Gemini Calls</span>
          <span className="metric-value">0</span>
        </div>
      </div>
    );
  }

  const {
    total_prompt_tokens = 0,
    cached_tokens = 0,
    fresh_tokens = 0,
    savings_percentage = 0.0,
    call_count = 0,
  } = tokenSummary;

  return (
    <div className="token-banner">
      <div className="metric-item">
        <span className="metric-label">Total Requested Tokens</span>
        <span className="metric-value">{total_prompt_tokens.toLocaleString()}</span>
      </div>
      <div className="metric-item">
        <span className="metric-label">Tokens Served from Cache</span>
        <span className="metric-value">{cached_tokens.toLocaleString()}</span>
      </div>
      <div className="metric-item">
        <span className="metric-label">Fresh Tokens Billed</span>
        <span className="metric-value">{fresh_tokens.toLocaleString()}</span>
      </div>
      <div className="metric-item">
        <span className="metric-label">Token Savings</span>
        <span className="metric-value highlight">{savings_percentage.toFixed(1)}%</span>
      </div>
      <div className="metric-item">
        <span className="metric-label">Total API Calls</span>
        <span className="metric-value">{call_count}</span>
      </div>
    </div>
  );
}
