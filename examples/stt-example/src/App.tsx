import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  FormControl,
  IconButton,
  InputLabel,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Paper,
  Select,
  Stack,
  Tooltip,
  Typography,
} from "@mui/material";
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
import {
  MdCheckCircle,
  MdDelete,
  MdDownload,
  MdError,
  MdMic,
  MdRefresh,
  MdStar,
  MdStarBorder,
  MdStop,
} from "react-icons/md";

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

const permColor = (status: string) =>
  status === "granted" ? "success" : status === "denied" ? "error" : "warning";

export default function App() {
  // Recognition state
  // Empty string = "auto-detect" (no language sent → Whisper detects from
  // audio). Guessing from `navigator.language` was misleading on browsers
  // that report locales Whisper doesn't actually transcribe well.
  const [language, setLanguage] = useState<string>("");
  const [recognitionState, setRecognitionState] = useState<RecognitionState>("idle");
  const [transcript, setTranscript] = useState<string>("");
  const [results, setResults] = useState<ResultRow[]>([]);
  /** Object URL of the most recent recording for the top-level player. */
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
  const [busy, setBusy] = useState(false); // start/stop button spinner

  const resultsEndRef = useRef<HTMLDivElement>(null);
  /** Holds all Blob object URLs so we can revoke them on clear/unmount. */
  const audioUrlsRef = useRef<string[]>([]);

  // Revoke all Blob URLs created so far.
  const revokeAllAudioUrls = useCallback(() => {
    for (const url of audioUrlsRef.current) URL.revokeObjectURL(url);
    audioUrlsRef.current = [];
  }, []);

  // Revoke on unmount.
  useEffect(() => revokeAllAudioUrls, [revokeAllAudioUrls]);

  useEffect(() => {
    resultsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [results]);
  const checkAvailability = useCallback(async () => {
    try {
      const result = await sttIsAvailable();
      setIsAvailable(result.available);
      setAvailabilityReason(result.reason ?? null);
      // The "no model installed" reason is expected on first run; the
      // model manager card already speaks to it, so don't surface a red
      // error toast for it.
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
      setPermission({
        microphone: perm.microphone,
        speechRecognition: perm.speechRecognition,
      });
    } catch (err) {
      if (!String(err).includes("Plugin not found")) {
        setError(`Failed to check permission: ${err}`);
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const pending: Array<Promise<unknown>> = [];
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
      pending.push(
        p.then((fn) => {
          if (cancelled) callUnlisten(fn);
          else unlistens.push(() => callUnlisten(fn));
        }),
      );
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
        // Convert base64 WAV → Blob URL for in-app playback (desktop only).
        let audioUrl: string | undefined;
        if (result.audioData) {
          try {
            const bytes = Uint8Array.from(atob(result.audioData), (c) =>
              c.charCodeAt(0),
            );
            audioUrl = URL.createObjectURL(
              new Blob([bytes], { type: "audio/wav" }),
            );
            audioUrlsRef.current.push(audioUrl);
            setLatestAudioUrl(audioUrl);
          } catch {
            // Non-fatal: just won't show a player for this result.
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
      // Anything still in-flight will be cleaned up when its promise
      // resolves (see the `cancelled` check inside `track`).
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // First availability probe.
  useEffect(() => {
    checkAvailability();
  }, [checkAvailability]);

  // After availability resolves, always load the model catalogue (the
  // user may need to install one) and load languages/permissions when
  // the plugin is actually usable.
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
      setPermission({
        microphone: perm.microphone,
        speechRecognition: perm.speechRecognition,
      });
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
    setDownloadProgress({
      status: "downloading",
      modelId: id,
      model: id,
      progress: 0,
    });
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
      await startListening({
        // `undefined` → backend uses Whisper's auto-detect (LANG_AUTO).
        language: language || undefined,
      });
      // recognitionState will flip to "listening" via the state listener.
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
    <Container maxWidth="md" sx={{ py: { xs: 2, sm: 3, md: 4 } }}>
      {/* Header ---------------------------------------------------------- */}
      <Paper
        elevation={0}
        sx={{
          p: { xs: 2, sm: 3 },
          mb: { xs: 2, sm: 3 },
          borderRadius: 2,
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          color: "white",
        }}
      >
        <Typography
          variant="h4"
          component="h1"
          sx={{
            fontSize: { xs: "1.5rem", sm: "2rem", md: "2.125rem" },
            fontWeight: 700,
            mb: 1,
          }}
        >
          🎤 Speech-to-Text Example (Whisper)
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.9 }}>
          Whisper is <strong>record-then-transcribe</strong>: press Start, speak your sentence,
          then press Stop. The model runs locally and emits the final transcript a moment later.
        </Typography>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {info && (
        <Alert severity="info" sx={{ mb: 2 }} onClose={() => setInfo(null)}>
          {info}
        </Alert>
      )}

      <Stack spacing={{ xs: 2, sm: 3 }}>
        {/* Download progress ------------------------------------------- */}
        {downloadProgress && (
          <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <CircularProgress
                size={24}
                variant={(downloadProgress.progress ?? 0) > 0 ? "determinate" : "indeterminate"}
                value={downloadProgress.progress ?? 0}
              />
              <Box sx={{ flex: 1 }}>
                <Typography>
                  {downloadProgress.status === "downloading"
                    ? `Downloading ${downloadProgress.modelId ?? downloadProgress.model}… ${downloadProgress.progress ?? 0}%`
                    : downloadProgress.status === "complete"
                      ? `${downloadProgress.modelId ?? downloadProgress.model} ready!`
                      : `Error downloading ${downloadProgress.model}`}
                </Typography>
                <Box
                  sx={{
                    mt: 1,
                    height: 4,
                    bgcolor: "rgba(0,0,0,0.1)",
                    borderRadius: 2,
                  }}
                >
                  <Box
                    sx={{
                      width: `${downloadProgress.progress ?? 0}%`,
                      height: "100%",
                      bgcolor: downloadProgress.status === "error" ? "#ef4444" : "#22c55e",
                      borderRadius: 2,
                      transition: "width 0.3s",
                    }}
                  />
                </Box>
              </Box>
            </Box>
          </Paper>
        )}

        {/* Availability  */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Stack spacing={1.5}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              {isAvailable === null ? (
                <CircularProgress size={20} />
              ) : isAvailable ? (
                <MdCheckCircle color="#22c55e" size={24} />
              ) : (
                <MdError color="#ef4444" size={24} />
              )}
              <Typography variant="h6">
                {isAvailable === null
                  ? "Checking availability…"
                  : isAvailable
                    ? "STT Available"
                    : hasInstalledModel
                      ? "STT Not Available"
                      : "Install a Whisper model to enable STT"}
              </Typography>
            </Box>
            {availabilityReason && (
              <Alert severity={NO_MODEL_HINT.test(availabilityReason) ? "info" : "warning"}>
                {availabilityReason}
              </Alert>
            )}
            {isAvailable === false && (
              <Button
                variant="outlined"
                onClick={checkAvailability}
                startIcon={<MdRefresh />}
                sx={{ alignSelf: "flex-start" }}
              >
                Recheck
              </Button>
            )}
          </Stack>
        </Paper>

        {/* Whisper model manager  */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              mb: 1.5,
              flexWrap: "wrap",
              gap: 1,
            }}
          >
            <Typography variant="h6">Whisper Models</Typography>
            <Stack direction="row" spacing={1} alignItems="center">
              <Tooltip title="Refresh catalogue">
                <IconButton size="small" onClick={() => refreshModels()}>
                  <MdRefresh />
                </IconButton>
              </Tooltip>
              <Button
                size="small"
                variant={showAdvanced ? "contained" : "outlined"}
                onClick={() => setShowAdvanced((v) => !v)}
              >
                {showAdvanced ? "Hide advanced" : "Show advanced"}
              </Button>
            </Stack>
          </Box>

          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
            Disk usage: {formatBytes(totalDiskBytes)} · System RAM:{" "}
            {systemMemoryMb > 0 ? `${systemMemoryMb} MB` : "unknown"}
          </Typography>

          {models.length === 0 ? (
            <Typography color="text.secondary" sx={{ py: 2 }}>
              Loading model catalogue…
            </Typography>
          ) : (
            <List disablePadding>
              {models.map((m, idx) => {
                const isInstalling =
                  installingId === m.id ||
                  (downloadProgress?.modelId === m.id &&
                    downloadProgress.status === "downloading");
                return (
                  <Box key={m.id}>
                    {idx > 0 && <Divider component="li" />}
                    <ListItem
                      sx={{ flexWrap: "wrap", gap: 1, py: 1.5 }}
                      secondaryAction={
                        <Stack direction="row" spacing={0.5}>
                          {m.installed ? (
                            <>
                              <Tooltip title={m.active ? "Active model" : "Make active"}>
                                <span>
                                  <IconButton
                                    size="small"
                                    onClick={() => handleSetActive(m.id)}
                                    disabled={m.active || isListening}
                                    color={m.active ? "primary" : "default"}
                                  >
                                    {m.active ? <MdStar /> : <MdStarBorder />}
                                  </IconButton>
                                </span>
                              </Tooltip>
                              <Tooltip title="Remove">
                                <span>
                                  <IconButton
                                    size="small"
                                    onClick={() => handleRemoveModel(m.id)}
                                    disabled={isListening}
                                  >
                                    <MdDelete />
                                  </IconButton>
                                </span>
                              </Tooltip>
                            </>
                          ) : (
                            <Tooltip
                              title={
                                m.fitsInMemory
                                  ? `Download ${m.sizeMb} MB`
                                  : `Needs ~${m.requiredMemoryMb} MB RAM`
                              }
                            >
                              <span>
                                <IconButton
                                  size="small"
                                  onClick={() => handleInstallModel(m.id)}
                                  disabled={!m.fitsInMemory || !!installingId || isListening}
                                >
                                  {isInstalling ? (
                                    <CircularProgress size={18} />
                                  ) : (
                                    <MdDownload />
                                  )}
                                </IconButton>
                              </span>
                            </Tooltip>
                          )}
                        </Stack>
                      }
                    >
                      <ListItemText
                        primary={
                          <Stack
                            direction="row"
                            spacing={1}
                            alignItems="center"
                            flexWrap="wrap"
                          >
                            <Typography component="span" sx={{ fontWeight: 600 }}>
                              {m.displayName}
                            </Typography>
                            {m.recommended && (
                              <Chip label="recommended" size="small" color="primary" />
                            )}
                            {m.active && <Chip label="active" size="small" color="success" />}
                            {m.installed && !m.active && (
                              <Chip label="installed" size="small" />
                            )}
                            {m.advanced && (
                              <Chip label="advanced" size="small" variant="outlined" />
                            )}
                          </Stack>
                        }
                        secondary={
                          <Stack direction="row" spacing={1.5} flexWrap="wrap" sx={{ mt: 0.5 }}>
                            <Typography
                              component="span"
                              variant="caption"
                              color="text.secondary"
                            >
                              {m.sizeMb} MB
                            </Typography>
                            <Typography
                              component="span"
                              variant="caption"
                              color="text.secondary"
                            >
                              ~{m.requiredMemoryMb} MB RAM
                            </Typography>
                            <Typography
                              component="span"
                              variant="caption"
                              color="text.secondary"
                            >
                              {m.tier}
                            </Typography>
                            <Typography
                              component="span"
                              variant="caption"
                              color="text.secondary"
                            >
                              {m.language ?? "multilingual"}
                            </Typography>
                            {!m.fitsInMemory && !m.installed && (
                              <Typography component="span" variant="caption" color="error">
                                Not enough RAM
                              </Typography>
                            )}
                          </Stack>
                        }
                      />
                    </ListItem>
                  </Box>
                );
              })}
            </List>
          )}
        </Paper>

        {/* Permissions  */}
        {permission && (
          <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
            <Typography variant="h6" sx={{ mb: 1.5 }}>
              Permissions
            </Typography>
            <Stack spacing={1}>
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                <Chip
                  label={`Microphone: ${permission.microphone}`}
                  color={permColor(permission.microphone)}
                  size="small"
                />
                <Chip
                  label={`Speech Recognition: ${permission.speechRecognition}`}
                  color={permColor(permission.speechRecognition)}
                  size="small"
                />
              </Box>
              {permission.microphone !== "granted" && (
                <Button
                  variant="contained"
                  size="small"
                  onClick={handleRequestPermission}
                  sx={{ alignSelf: "flex-start" }}
                >
                  Request Permission
                </Button>
              )}
            </Stack>
          </Paper>
        )}

        {/* Configuration  */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Typography variant="h6" sx={{ mb: 1.5 }}>
            Configuration
          </Typography>
          <FormControl
            size="small"
            fullWidth
            disabled={isListening || isProcessing}
            sx={{ maxWidth: 360 }}
          >
            <InputLabel id="stt-language-label">Language</InputLabel>
            <Select
              labelId="stt-language-label"
              label="Language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              MenuProps={{ PaperProps: { sx: { maxHeight: 360 } } }}
            >
              <MenuItem value="">
                <em>Auto-detect</em>
              </MenuItem>
              {availableLanguages.map((lang) => (
                <MenuItem key={lang.code} value={lang.code}>
                  {lang.name}{" "}
                  <Typography
                    component="span"
                    variant="caption"
                    color="text.secondary"
                    sx={{ ml: 1 }}
                  >
                    ({lang.code})
                  </Typography>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
            {availableLanguages.length > 0 &&
              `${availableLanguages.length} languages available. `}
            Whisper has a single <code>pt</code> for all Portuguese variants (Brazilian and
            European). Its <code>br</code> code is <strong>Breton</strong>, not Brazilian — when
            in doubt, leave on <em>Auto-detect</em>.
          </Typography>
        </Paper>

        {/* Controls  */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={1.5}
            sx={{ justifyContent: "center" }}
          >
            {!isListening ? (
              <Button
                variant="contained"
                size="large"
                onClick={handleStart}
                disabled={busy || !canListen}
                startIcon={busy ? <CircularProgress size={20} /> : <MdMic />}
                sx={{
                  bgcolor: "#ef4444",
                  "&:hover": { bgcolor: "#dc2626" },
                }}
              >
                Start Listening
              </Button>
            ) : (
              <Button
                variant="contained"
                size="large"
                onClick={handleStop}
                disabled={busy}
                startIcon={busy ? <CircularProgress size={20} /> : <MdStop />}
                sx={{
                  bgcolor: "#dc2626",
                  "&:hover": { bgcolor: "#b91c1c" },
                }}
              >
                Stop & Transcribe
              </Button>
            )}
            <Button
              variant="outlined"
              onClick={handleClear}
              disabled={isListening || isProcessing}
              startIcon={<MdRefresh />}
            >
              Clear
            </Button>
          </Stack>

          {!canListen && !isListening && !isProcessing && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: "block", textAlign: "center", mt: 1 }}
            >
              {!activeModelId
                ? "Install and activate a Whisper model above to enable listening."
                : isAvailable === false
                  ? "STT plugin not available."
                  : "Preparing…"}
            </Typography>
          )}
        </Paper>

        {/* Transcription  */}
        <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              mb: 1.5,
            }}
          >
            <Typography variant="h6">Transcription</Typography>
            {isListening && (
              <Stack direction="row" spacing={1} alignItems="center" sx={{ color: "#ef4444" }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    bgcolor: "#ef4444",
                    animation: "pulse 1s infinite",
                    "@keyframes pulse": {
                      "0%, 100%": { opacity: 1 },
                      "50%": { opacity: 0.3 },
                    },
                  }}
                />
                <Typography variant="body2">Recording…</Typography>
              </Stack>
            )}
            {isProcessing && (
              <Stack direction="row" spacing={1} alignItems="center">
                <CircularProgress size={16} />
                <Typography variant="body2">Transcribing…</Typography>
              </Stack>
            )}
          </Box>

          <Box
            sx={{
              minHeight: 100,
              p: 2,
              borderRadius: 1,
              bgcolor: "action.hover",
              fontSize: "1.05rem",
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {transcript || (
              <Typography color="text.secondary">
                {isListening
                  ? "Listening… speak now, then press Stop."
                  : isProcessing
                    ? "Whisper is decoding your audio…"
                    : "Press Start, speak a sentence, then press Stop. The transcript will appear here."}
              </Typography>
            )}
          </Box>
          {latestAudioUrl && (
            <Box sx={{ mt: 1.5 }}>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: "block" }}>
                Last recording
              </Typography>
              <Box
                component="audio"
                controls
                src={latestAudioUrl}
                sx={{ width: "100%", display: "block" }}
              />
            </Box>
          )}
        </Paper>

        {/* History ---------------------------------------------------- */}
        {results.length > 0 && (
          <Paper sx={{ p: { xs: 1.5, sm: 2 } }}>
            <Typography variant="h6" sx={{ mb: 1.5 }}>
              Results History ({results.length})
            </Typography>
            <List
              sx={{
                maxHeight: 300,
                overflow: "auto",
                bgcolor: "action.hover",
                borderRadius: 1,
              }}
            >
              {results.map((r, idx) => (
                <ListItem key={idx} divider>
                  <ListItemText
                    primary={r.text}
                    secondary={
                      <Stack spacing={1} sx={{ mt: 1 }}>
                        <Stack direction="row" spacing={1}>
                          {r.confidence !== undefined && (
                            <Chip label={`${(r.confidence * 100).toFixed(0)}%`} size="small" />
                          )}
                          <Chip label={r.timestamp.toLocaleTimeString()} size="small" />
                        </Stack>
                        {r.audioUrl && (
                          <Box
                            component="audio"
                            controls
                            src={r.audioUrl}
                            sx={{ width: "100%", display: "block" }}
                          />
                        )}
                      </Stack>
                    }
                  />
                </ListItem>
              ))}
              <div ref={resultsEndRef} />
            </List>
          </Paper>
        )}
      </Stack>
    </Container>
  );
}
