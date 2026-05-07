import { invoke, PluginListener, addPluginListener } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

export interface ListenConfig {
  /** Language code for recognition (e.g., "en-US", "pt-BR") */
  language?: string;
  /** Whether to return interim (partial) results */
  interimResults?: boolean;
  /** Whether to continue listening after getting a result */
  continuous?: boolean;
  /** Maximum duration to listen in milliseconds (0 = no limit) */
  maxDuration?: number;
  /** Use on-device recognition only (iOS 13+, no network required)
   * When true, recognition works offline but may be less accurate.
   * Falls back to server if on-device not available for the language.
   */
  onDevice?: boolean;
}

export type RecognitionState = "idle" | "listening" | "processing";

export interface RecognitionResult {
  transcript: string;
  isFinal: boolean;
  confidence?: number;
  audioData?: string;
}

export type SttErrorCode =
  | "NONE"
  | "NOT_AVAILABLE"
  | "PERMISSION_DENIED"
  | "SPEECH_PERMISSION_DENIED"
  | "NETWORK_ERROR"
  | "AUDIO_ERROR"
  | "TIMEOUT"
  | "NO_SPEECH"
  | "LANGUAGE_NOT_SUPPORTED"
  | "CANCELLED"
  | "ALREADY_LISTENING"
  | "NOT_LISTENING"
  | "BUSY"
  | "UNKNOWN";

export interface SttError {
  /** Error code for programmatic handling */
  code: SttErrorCode;
  /** Human-readable error message */
  message: string;
  /** Platform-specific error details */
  details?: string;
}

export interface StateChangeEvent {
  state: RecognitionState;
}

export interface SupportedLanguage {
  code: string;
  name: string;
  installed?: boolean;
}

export interface AvailabilityResponse {
  available: boolean;
  reason?: string;
}

export interface SupportedLanguagesResponse {
  languages: SupportedLanguage[];
}

export type PermissionStatus = "granted" | "denied" | "unknown";

export interface PermissionResponse {
  microphone: PermissionStatus;
  speechRecognition: PermissionStatus;
}

/** Metadata for a single Whisper model in the catalogue. */
export interface WhisperModelInfo {
  /** Stable identifier (`tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v3`). */
  id: string;
  /** Human-readable name shown in the model manager. */
  displayName: string;
  /** Approximate on-disk size in megabytes. */
  sizeMb: number;
  /** Approximate working-set memory in megabytes (whisper.cpp's published "required memory"). */
  requiredMemoryMb: number;
  /** Whether the binary is currently present in the app data dir. */
  installed: boolean;
  /** Whether this model is the one `startListening` will load. */
  active: boolean;
  /** Suggested default for first-time users (exactly one model carries this flag). */
  recommended: boolean;
  /** Short qualitative tier label (`fastest`, `balanced`, `most accurate`, ...). */
  tier: string;
  /** `"en"` for English-optimised variants (`*.en`); `null` for multilingual models. */
  language?: string | null;
  /** `false` when the local machine doesn't have enough RAM/VRAM to load this model. */
  fitsInMemory: boolean;
  /** Power-user model (currently the `large` family); hidden unless `includeAdvanced` is true. */
  advanced: boolean;
}

export interface WhisperModelsResponse {
  /** Catalogue ordered from smallest to largest. */
  models: WhisperModelInfo[];
  /** Currently active model id (`null` if none installed yet). */
  active?: string | null;
  /** Total bytes occupied by every installed model. */
  totalDiskBytes: number;
  /** Total physical RAM (in MB) the host machine reports. */
  systemMemoryMb: number;
}

/** Payload of `stt://download-progress` events. */
export interface DownloadProgressEvent {
  /** `downloading` | `complete` | `error` */
  status: "downloading" | "complete" | "error";
  /** Stable model identifier (e.g. `tiny.en`). */
  modelId?: string;
  /** Underlying file name (e.g. `ggml-tiny.en.bin`). */
  model: string;
  /** 0–100 percent. Only meaningful while `status === "downloading"` or `complete`. */
  progress?: number;
  /** Bytes downloaded so far (downloading only). */
  downloaded?: number;
  /** Total bytes if the server returned `Content-Length` (downloading only). */
  total?: number;
  /** Error message (error only). */
  message?: string;
}

export async function startListening(config?: ListenConfig): Promise<void> {
  await invoke("plugin:stt|start_listening", { config: config || {} });
}

export async function stopListening(): Promise<void> {
  await invoke("plugin:stt|stop_listening");
}

export async function isAvailable(): Promise<AvailabilityResponse> {
  return await invoke("plugin:stt|is_available");
}

export async function getSupportedLanguages(): Promise<SupportedLanguagesResponse> {
  return await invoke("plugin:stt|get_supported_languages");
}

export async function checkPermission(): Promise<PermissionResponse> {
  return await invoke("plugin:stt|check_permission");
}

export async function requestPermission(): Promise<PermissionResponse> {
  return await invoke("plugin:stt|request_permission");
}

export async function listModels(includeAdvanced = false): Promise<WhisperModelsResponse> {
  return await invoke("plugin:stt|list_models", { includeAdvanced });
}

export async function installModel(id: string): Promise<void> {
  await invoke("plugin:stt|install_model", { id });
}

export async function removeModel(id: string): Promise<void> {
  await invoke("plugin:stt|remove_model", { id });
}

export async function setActiveModel(id: string): Promise<void> {
  await invoke("plugin:stt|set_active_model", { id });
}

export async function onDownloadProgress(
  handler: (event: DownloadProgressEvent) => void,
): Promise<UnlistenFn> {
  return await listen<DownloadProgressEvent>("stt://download-progress", (event) => {
    handler(event.payload);
  });
}

export async function onResult(
  handler: (result: RecognitionResult) => void,
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<RecognitionResult>("stt", "result", handler);
  }

  const unlisten = await listen<RecognitionResult>("plugin:stt:result", (event) => {
    handler(event.payload);
  });
  return unlisten;
}

export async function onStateChange(
  handler: (event: StateChangeEvent) => void,
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<StateChangeEvent>("stt", "stateChange", handler);
  }

  const unlisten = await listen<StateChangeEvent>("plugin:stt:stateChange", (event) => {
    handler(event.payload);
  });
  return unlisten;
}

export async function onError(
  handler: (error: SttError) => void,
): Promise<PluginListener | UnlistenFn> {
  const isMobile = isMobilePlatform();

  if (isMobile) {
    return await addPluginListener<SttError>("stt", "error", handler);
  }

  return await listen<SttError>("plugin:stt:error", (event) => {
    handler(event.payload);
  });
}

function isMobilePlatform(): boolean {
  // Check for mobile-specific Tauri internals and Android WebView
  const w = window as any;

  // Check Tauri's internal platform detection first
  const platform = w.__TAURI_INTERNALS__?.plugins?.os?.platform;
  if (platform === "android" || platform === "ios") {
    return true;
  }

  // Check for Android WebView
  if (w.Android) {
    return true;
  }

  // Check for iOS-specific WebKit (but not macOS)
  // iOS has webkit.messageHandlers AND navigator.userAgent contains "iPhone" or "iPad"
  if (w.webkit?.messageHandlers) {
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("iphone") || ua.includes("ipad") || ua.includes("ipod")) {
      return true;
    }
  }

  // Default to desktop
  return false;
}
