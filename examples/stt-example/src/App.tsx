import { useCallback, useEffect, useRef, useState } from "react";
import {
  checkPermission as sttCheckPermission,
  getSupportedLanguages,
  installModel,
  isAvailable as sttIsAvailable,
  listModels,
  onDownloadProgress,
  onError,
  onResult,
  onStateChange,
  removeModel,
  requestPermission as sttRequestPermission,
  setActiveModel,
  startListening,
  stopListening,
  type DownloadProgressEvent,
  type RecognitionState,
  type SupportedLanguage,
  type WhisperModelInfo,
} from "tauri-plugin-stt-api";
import "./App.css";

const NO_MODEL_HINT = /no whisper model/i;

type PermState = { microphone: string; speechRecognition: string } | null;

type ResultRow = {
  text: string;
  confidence?: number;
  timestamp: Date;
  audioUrl?: string;
};

const formatBytes = (bytes: number) => {
  if (bytes <= 0) return "0 MB";
  const mb = bytes / (1024 * 1024);
  return mb < 1024 ? `${mb.toFixed(1)} MB` : `${(mb / 1024).toFixed(2)} GB`;
};

export default function App() {
  const [language, setLanguage] = useState<string>("");
  const [recognitionState, setRecognitionState] = useState<RecognitionState>("idle");
  const [transcript, setTranscript] = useState<string>("");
  const [results, setResults] = useState<ResultRow[]>([]);
  const [latestAudioUrl, setLatestAudioUrl] = useState<string | null>(null);

  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);
  const [availabilityReason, setAvailabilityReason] = useState<string | null>(null);
  const [permission, setPermission] = useState<PermState>(null);
  const [availableLanguages, setAvailableLanguages] = useState<SupportedLanguage[]>([]);

  const [models, setModels] = useState<WhisperModelInfo[]>([]);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [totalDiskBytes, setTotalDiskBytes] = useState(0);
  const [systemMemoryMb, setSystemMemoryMb] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [installingId, setInstallingId] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgressEvent | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const resultsEndRef = useRef<HTMLDivElement>(null);
  const audioUrlsRef = useRef<string[]>([]);

  const revokeAllAudioUrls = useCallback(() => {
    for (const url of audioUrlsRef.current) URL.revokeObjectURL(url);
    audioUrlsRef.current = [];
  }, []);

  useEffect(() => revokeAllAudioUrls, [revokeAllAudioUrls]);

  useEffect(() => {
    resultsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [results]);

  const checkAvailability = useCallback(async () => {
    try {
      const result = await sttIsAvailable();
      setIsAvailable(result.available);
      setAvailabilityReason(result.reason ?? null);
      if (!result.available && result.reason && !NO_MODEL_HINT.test(result.reason)) {
        setError(result.reason);
      }
    } catch (err) {
      const msg = String(err);
      if (msg.includes("Plugin not found")) {
        setError("STT plugin is not enabled. Build with --features stt to enable it.");
        setAvailabilityReason("Plugin not compiled with STT support");
      } else {
        setError(`Failed to check availability: ${err}`);
      }
      setIsAvailable(false);
    }
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      const resp = await listModels(showAdvanced);
      setModels(resp.models);
      setActiveModelId(resp.active ?? null);
      setTotalDiskBytes(resp.totalDiskBytes);
      setSystemMemoryMb(resp.systemMemoryMb);
    } catch (err) {
      if (!String(err).includes("Plugin not found")) {
        setError(`Failed to load models: ${err}`);
      }
    }
  }, [showAdvanced]);

  const loadLanguages = useCallback(async () => {
    try {
      const resp = await getSupportedLanguages();
      setAvailableLanguages(resp.languages);
    } catch (err) {
      if (!String(err).includes("Plugin not found")) {
        setError(`Failed to load languages: ${err}`);
      }
    }
  }, []);

  const checkPerm = useCallback(async () => {
    try {
      const perm = await sttCheckPermission();
      setPermission({ microphone: perm.microphone, speechRecognition: perm.speechRecognition });
    } catch (err) {
      if (!String(err).includes("Plugin not found")) {
        setError(`Failed to check permission: ${err}`);
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const unlistens: Array<() => void> = [];

    const callUnlisten = (l: unknown) => {
      if (!l) return;
      if (typeof l === "function") (l as () => void)();
      else if (
        typeof l === "object" &&
        l !== null &&
        typeof (l as { unregister?: () => void }).unregister === "function"
      )
        (l as { unregister: () => void }).unregister();
    };

    const track = <T,>(p: Promise<T>) => {
      p.then((fn) => {
        if (cancelled) callUnlisten(fn);
        else unlistens.push(() => callUnlisten(fn));
      });
    };

    track(
      onResult((result) => {
        const text = result.transcript.trim();
        if (!text) {
          setInfo(
            "No speech detected. Try recording a longer clip — or check the language is correct.",
          );
          setTimeout(() => setInfo(null), 4000);
          return;
        }
        setTranscript((prev) => (prev ? `${prev} ${text}` : text));
        let audioUrl: string | undefined;
        if (result.audioData) {
          try {
            const bytes = Uint8Array.from(atob(result.audioData), (c) => c.charCodeAt(0));
            audioUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
            audioUrlsRef.current.push(audioUrl);
            setLatestAudioUrl(audioUrl);
          } catch {
            // Non-fatal: playback unavailable for this result.
          }
        }
        setResults((prev) => [
          ...prev,
          { text, confidence: result.confidence, timestamp: new Date(), audioUrl },
        ]);
      }),
    );

    track(
      onStateChange((event) => {
        setRecognitionState(event.state);
      }),
    );

    track(
      onError((err) => {
        console.error("STT error:", err);
        setError(`STT Error: ${err.message ?? err.code ?? String(err)}`);
        setBusy(false);
      }),
    );

    track(
      onDownloadProgress((event) => {
        setDownloadProgress(event);
        if (event.status === "complete") {
          setInfo(`Model ${event.modelId ?? event.model} downloaded!`);
          setTimeout(() => {
            setDownloadProgress(null);
            setInfo(null);
          }, 1500);
          refreshModels();
          checkAvailability();
        } else if (event.status === "error") {
          setError(`Download failed: ${event.message ?? "unknown error"}`);
          setDownloadProgress(null);
        }
      }),
    );

    return () => {
      cancelled = true;
      for (const u of unlistens) u();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    checkAvailability();
  }, [checkAvailability]);

  useEffect(() => {
    if (isAvailable === null) return;
    refreshModels();
    if (isAvailable) {
      loadLanguages();
      checkPerm();
    }
  }, [isAvailable, refreshModels, loadLanguages, checkPerm]);

  const handleRequestPermission = async () => {
    try {
      const perm = await sttRequestPermission();
      setPermission({ microphone: perm.microphone, speechRecognition: perm.speechRecognition });
      if (perm.microphone === "granted") {
        setInfo("Permission granted!");
        setTimeout(() => setInfo(null), 2000);
      }
    } catch (err) {
      setError(`Failed to request permission: ${err}`);
    }
  };

  const handleInstallModel = async (id: string) => {
    setError(null);
    setInstallingId(id);
    setDownloadProgress({ status: "downloading", modelId: id, model: id, progress: 0 });
    try {
      await installModel(id);
    } catch (err) {
      setError(`Install failed: ${err}`);
      setDownloadProgress(null);
    } finally {
      setInstallingId(null);
    }
  };

  const handleRemoveModel = async (id: string) => {
    setError(null);
    try {
      await removeModel(id);
      setInfo(`Removed ${id}`);
      setTimeout(() => setInfo(null), 1500);
      await refreshModels();
      await checkAvailability();
    } catch (err) {
      setError(`Remove failed: ${err}`);
    }
  };

  const handleSetActive = async (id: string) => {
    setError(null);
    try {
      await setActiveModel(id);
      setActiveModelId(id);
      setInfo(`${id} is now the active model`);
      setTimeout(() => setInfo(null), 1500);
      await refreshModels();
      await checkAvailability();
    } catch (err) {
      setError(`Set active failed: ${err}`);
    }
  };

  const handleStart = async () => {
    setError(null);
    setBusy(true);
    try {
      await startListening({ language: language || undefined });
      setInfo("Listening… speak, then press Stop.");
      setTimeout(() => setInfo(null), 2500);
    } catch (err) {
      const msg = String(err);
      if (NO_MODEL_HINT.test(msg)) {
        setError(
          "No Whisper model installed yet. Install one from the Whisper Models panel above.",
        );
      } else if (msg.includes("permission")) {
        setError(`Permission required: ${err}`);
      } else {
        setError(`Failed to start listening: ${err}`);
      }
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      await stopListening();
    } catch (err) {
      setError(`Failed to stop listening: ${err}`);
    } finally {
      setBusy(false);
    }
  };

  const handleClear = () => {
    setTranscript("");
    setResults([]);
    revokeAllAudioUrls();
    setLatestAudioUrl(null);
  };

  const hasInstalledModel = models.some((m) => m.installed);
  const isListening = recognitionState === "listening";
  const isProcessing = recognitionState === "processing";
  const canListen = isAvailable === true && !!activeModelId && !installingId && !isProcessing;

  return (
    <div className="page">
      <header className="header">
        <div className="header-inner">
          <h1 className="header-title">Speech-to-Text</h1>
          <p className="header-subtitle">
            Tauri Plugin STT — Whisper runs locally, record then transcribe
          </p>
        </div>
      </header>

      <main className="main">
        {error && (
          <div className="alert alert-error">
            <span>{error}</span>
            <button className="alert-close" onClick={() => setError(null)}>
              ×
            </button>
          </div>
        )}
        {info && (
          <div className="alert alert-info">
            <span>{info}</span>
            <button className="alert-close" onClick={() => setInfo(null)}>
              ×
            </button>
          </div>
        )}

        {/* Download progress */}
        {downloadProgress && (
          <div className="card">
            <div className="progress-header">
              {downloadProgress.status === "downloading" && (
                <span className="spinner" aria-label="downloading" />
              )}
              <span>
                {downloadProgress.status === "downloading"
                  ? `Downloading ${downloadProgress.modelId ?? downloadProgress.model}… ${downloadProgress.progress ?? 0}%`
                  : downloadProgress.status === "complete"
                    ? `${downloadProgress.modelId ?? downloadProgress.model} ready!`
                    : `Error downloading ${downloadProgress.model}`}
              </span>
            </div>
            <div className="progress-bar">
              <div
                className={`progress-fill${downloadProgress.status === "error" ? " progress-fill--error" : ""}`}
                style={{ width: `${downloadProgress.progress ?? 0}%` }}
              />
            </div>
          </div>
        )}

        {/* Availability */}
        <div className="card">
          <div className="status-row">
            <span
              className={`status-dot ${
                isAvailable === null
                  ? "status-dot--pending"
                  : isAvailable
                    ? "status-dot--ok"
                    : "status-dot--error"
              }`}
            />
            <span className="card-title">
              {isAvailable === null
                ? "Checking availability…"
                : isAvailable
                  ? "STT Available"
                  : hasInstalledModel
                    ? "STT Not Available"
                    : "Install a Whisper model to enable STT"}
            </span>
          </div>
          {availabilityReason && (
            <p
              className={`avail-reason ${
                NO_MODEL_HINT.test(availabilityReason) ? "avail-reason--info" : "avail-reason--warn"
              }`}
            >
              {availabilityReason}
            </p>
          )}
          {isAvailable === false && (
            <button className="btn btn-secondary btn-sm" onClick={checkAvailability}>
              ↻ Recheck
            </button>
          )}
        </div>

        {/* Whisper Models */}
        <div className="card">
          <div className="card-row">
            <h2 className="card-title">Whisper Models</h2>
            <div className="card-row-actions">
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => refreshModels()}
                title="Refresh catalogue"
              >
                ↻
              </button>
              <button
                className={`btn btn-sm ${showAdvanced ? "btn-primary" : "btn-secondary"}`}
                onClick={() => setShowAdvanced((v) => !v)}
              >
                {showAdvanced ? "Hide advanced" : "Show advanced"}
              </button>
            </div>
          </div>

          <p className="meta-line">
            Disk: {formatBytes(totalDiskBytes)}
            {systemMemoryMb > 0 && ` · RAM: ${systemMemoryMb} MB`}
          </p>

          {models.length === 0 ? (
            <p className="empty">Loading model catalogue…</p>
          ) : (
            <div className="model-list">
              {models.map((m) => {
                const isInstalling =
                  installingId === m.id ||
                  (downloadProgress?.modelId === m.id &&
                    downloadProgress.status === "downloading");
                return (
                  <div key={m.id} className="model-item">
                    <div className="model-info">
                      <div className="model-name-row">
                        <span className="model-name">{m.displayName}</span>
                        {m.recommended && <span className="badge">recommended</span>}
                        {m.active && <span className="badge badge--green">active</span>}
                        {m.installed && !m.active && (
                          <span className="badge badge--gray">installed</span>
                        )}
                        {m.advanced && <span className="badge badge--outline">advanced</span>}
                      </div>
                      <div className="model-meta">
                        <span>{m.sizeMb} MB</span>
                        <span>~{m.requiredMemoryMb} MB RAM</span>
                        <span>{m.tier}</span>
                        <span>{m.language ?? "multilingual"}</span>
                        {!m.fitsInMemory && !m.installed && (
                          <span className="meta-warn">Not enough RAM</span>
                        )}
                      </div>
                    </div>
                    <div className="model-actions">
                      {m.installed ? (
                        <>
                          <button
                            className={`btn-icon${m.active ? " btn-icon--active" : ""}`}
                            onClick={() => handleSetActive(m.id)}
                            disabled={m.active || isListening}
                            title={m.active ? "Active model" : "Make active"}
                          >
                            {m.active ? "★" : "☆"}
                          </button>
                          <button
                            className="btn-icon"
                            onClick={() => handleRemoveModel(m.id)}
                            disabled={isListening}
                            title="Remove"
                          >
                            ✕
                          </button>
                        </>
                      ) : (
                        <button
                          className="btn-icon"
                          onClick={() => handleInstallModel(m.id)}
                          disabled={!m.fitsInMemory || !!installingId || isListening}
                          title={
                            m.fitsInMemory
                              ? `Download ${m.sizeMb} MB`
                              : `Needs ~${m.requiredMemoryMb} MB RAM`
                          }
                        >
                          {isInstalling ? <span className="spinner spinner--sm" /> : "↓"}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Permissions */}
        {permission && (
          <div className="card">
            <h2 className="card-title">Permissions</h2>
            <div className="perm-badges">
              <span
                className={`perm-badge perm-badge--${permission.microphone === "granted" ? "granted" : permission.microphone === "denied" ? "denied" : "prompt"}`}
              >
                Microphone: {permission.microphone}
              </span>
              <span
                className={`perm-badge perm-badge--${permission.speechRecognition === "granted" ? "granted" : permission.speechRecognition === "denied" ? "denied" : "prompt"}`}
              >
                Speech Recognition: {permission.speechRecognition}
              </span>
            </div>
            {permission.microphone !== "granted" && (
              <button
                className="btn btn-primary btn-sm"
                onClick={handleRequestPermission}
                style={{ alignSelf: "flex-start" }}
              >
                Request Permission
              </button>
            )}
          </div>
        )}

        {/* Configuration */}
        <div className="card">
          <h2 className="card-title">Configuration</h2>
          <div className="field">
            <label className="field-label" htmlFor="stt-language">
              Language
            </label>
            <div className="select-wrap">
              <select
                id="stt-language"
                className="select"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={isListening || isProcessing}
              >
                <option value="">Auto-detect</option>
                {availableLanguages.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name} ({lang.code})
                  </option>
                ))}
              </select>
              <svg
                className="select-chevron"
                width="12"
                height="12"
                viewBox="0 0 12 12"
                aria-hidden
              >
                <path
                  d="M2 4l4 4 4-4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />
              </svg>
            </div>
            <p className="field-hint">
              {availableLanguages.length > 0 &&
                `${availableLanguages.length} languages available. `}
              Whisper uses <code>pt</code> for all Portuguese variants. Its <code>br</code> code is
              Breton, not Brazilian — when in doubt, leave on Auto-detect.
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="card">
          <div className="actions">
            {!isListening ? (
              <button
                className="btn btn-record"
                onClick={handleStart}
                disabled={busy || !canListen}
              >
                {busy ? <span className="spinner spinner--white" /> : "●"} Start Listening
              </button>
            ) : (
              <button
                className="btn btn-record btn-record--stop"
                onClick={handleStop}
                disabled={busy}
              >
                {busy ? <span className="spinner spinner--white" /> : "■"} Stop & Transcribe
              </button>
            )}
            <button
              className="btn btn-secondary"
              onClick={handleClear}
              disabled={isListening || isProcessing}
            >
              Clear
            </button>
          </div>

          {!canListen && !isListening && !isProcessing && (
            <p className="controls-hint">
              {!activeModelId
                ? "Install and activate a Whisper model above to enable listening."
                : isAvailable === false
                  ? "STT plugin not available."
                  : "Preparing…"}
            </p>
          )}
        </div>

        {/* Transcription */}
        <div className="card">
          <div className="card-row">
            <h2 className="card-title">Transcription</h2>
            <div className="rec-status">
              {isListening && (
                <>
                  <span className="rec-dot" />
                  <span className="rec-label">Recording…</span>
                </>
              )}
              {isProcessing && (
                <>
                  <span className="spinner spinner--sm" />
                  <span className="rec-label">Transcribing…</span>
                </>
              )}
            </div>
          </div>

          <div className="transcript-box">
            {transcript || (
              <span className="transcript-placeholder">
                {isListening
                  ? "Listening… speak now, then press Stop."
                  : isProcessing
                    ? "Whisper is decoding your audio…"
                    : "Press Start, speak a sentence, then press Stop. The transcript will appear here."}
              </span>
            )}
          </div>

          {latestAudioUrl && (
            <div className="audio-player">
              <p className="field-label">Last recording</p>
              <audio controls src={latestAudioUrl} style={{ width: "100%", display: "block" }} />
            </div>
          )}
        </div>

        {/* Results history */}
        {results.length > 0 && (
          <div className="card">
            <h2 className="card-title">Results History ({results.length})</h2>
            <div className="results-list">
              {results.map((r, idx) => (
                <div key={idx} className="result-item">
                  <p className="result-text">{r.text}</p>
                  <div className="result-meta">
                    {r.confidence !== undefined && (
                      <span className="badge badge--gray">
                        {(r.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                    <span className="badge badge--gray">{r.timestamp.toLocaleTimeString()}</span>
                  </div>
                  {r.audioUrl && (
                    <audio
                      controls
                      src={r.audioUrl}
                      style={{ width: "100%", display: "block", marginTop: 8 }}
                    />
                  )}
                </div>
              ))}
              <div ref={resultsEndRef} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
