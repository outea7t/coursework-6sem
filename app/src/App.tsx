import { useState, useEffect, useRef, useCallback } from "react";
import Chat from "./components/Chat";
import SettingsModal from "./components/SettingsModal";
import type { Settings, Message, GenerationMessage, BridgeStatus, BridgeEvent, GenerateParams } from "./types";

const DEFAULT_SETTINGS: Settings = {
  steps: 30,
  guidance: 7.5,
  width: 1024,
  height: 1024,
  negative_prompt: "low quality, blurry, distorted, ugly, bad anatomy",
  seed: "",
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [bridgeStatus, setBridgeStatus] = useState<BridgeStatus>("loading");
  const [statusMessage, setStatusMessage] = useState("Запуск...");
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);

  const generationIdRef = useRef(0);

  useEffect(() => {
    if (!window.bridge) return;

    const unsubscribe = window.bridge.onEvent((event: BridgeEvent) => {
      switch (event.type) {
        case "status":
          setStatusMessage(event.message ?? "");
          break;

        case "loading_progress":
          setLoadingProgress(event.progress ?? 0);
          setStatusMessage(event.message ?? "");
          break;

        case "ready":
          setBridgeStatus("ready");
          setStatusMessage("");
          setLoadingProgress(100);
          break;

        case "generation_started":
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.type === "generation") {
              (last as GenerationMessage).totalSteps = event.total_steps ?? 0;
            }
            return updated;
          });
          break;

        case "progress":
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.type === "generation") {
              const gen = last as GenerationMessage;
              gen.currentStep = event.step ?? 0;
              gen.totalSteps = event.total ?? 0;
              if (event.imageData) {
                gen.currentImage = event.imageData;
              }
            }
            return [...updated];
          });
          break;

        case "generation_done":
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.type === "generation") {
              const gen = last as GenerationMessage;
              gen.isComplete = true;
              gen.elapsed = event.elapsed ?? null;
              if (event.imageData) {
                gen.currentImage = event.imageData;
              }
            }
            return [...updated];
          });
          setIsGenerating(false);
          break;

        case "error":
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.type === "generation" && !(last as GenerationMessage).isComplete) {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...(last as GenerationMessage),
                isComplete: true,
                error: event.message ?? "Unknown error",
              };
              return updated;
            }
            return [
              ...prev,
              {
                id: Date.now(),
                type: "generation" as const,
                prompt: "",
                currentStep: 0,
                totalSteps: 0,
                currentImage: null,
                isComplete: true,
                error: event.message ?? "Unknown error",
                elapsed: null,
              },
            ];
          });
          setIsGenerating(false);
          break;
      }
    });

    return unsubscribe;
  }, []);

  const handleSend = useCallback(
    (text: string) => {
      if (!text.trim() || isGenerating || bridgeStatus !== "ready") return;

      const userMsg: Message = {
        id: Date.now(),
        type: "user",
        text: text.trim(),
      };

      const genId = ++generationIdRef.current;
      const genMsg: GenerationMessage = {
        id: genId,
        type: "generation",
        prompt: text.trim(),
        currentStep: 0,
        totalSteps: settings.steps,
        currentImage: null,
        isComplete: false,
        error: null,
        elapsed: null,
      };

      setMessages((prev) => [...prev, userMsg, genMsg]);
      setIsGenerating(true);

      const params: GenerateParams = {
        prompt: text.trim(),
        negative_prompt: settings.negative_prompt,
        steps: settings.steps,
        guidance: settings.guidance,
        width: settings.width,
        height: settings.height,
      };
      if (settings.seed !== "") {
        params.seed = parseInt(settings.seed, 10);
      }

      window.bridge.generate(params);
    },
    [isGenerating, bridgeStatus, settings],
  );

  return (
    <div className="app">
      <div className="drag-region">
        <div className="title-bar">
          <span className="title-text">Diffusion Pipeline</span>
          <button
            className="settings-btn"
            onClick={() => setIsSettingsOpen(true)}
            title="Настройки"
          >
            <SettingsIcon />
          </button>
        </div>
      </div>

      {bridgeStatus === "loading" && (
        <div className="loading-overlay">
          <div className="loading-bar-wrap">
            <div className="loading-bar-fill" style={{ width: `${loadingProgress}%` }} />
          </div>
          <p className="loading-text">{statusMessage}</p>
          <p className="loading-percent">{loadingProgress}%</p>
        </div>
      )}

      <Chat
        messages={messages}
        onSend={handleSend}
        isGenerating={isGenerating}
        isReady={bridgeStatus === "ready"}
        onImageClick={setLightboxImage}
      />

      {lightboxImage && (
        <div className="lightbox" onClick={() => setLightboxImage(null)}>
          <img src={lightboxImage} alt="Preview" className="lightbox-image" />
        </div>
      )}

      {isSettingsOpen && (
        <SettingsModal
          settings={settings}
          onSave={(s: Settings) => {
            setSettings(s);
            setIsSettingsOpen(false);
          }}
          onClose={() => setIsSettingsOpen(false)}
        />
      )}
    </div>
  );
}

function SettingsIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}
