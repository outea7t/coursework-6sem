import TypewriterText from "./TypewriterText";
import type { GenerationMessage } from "../types";

interface GenerationProgressProps {
  message: GenerationMessage;
  onImageClick: (src: string) => void;
}

export default function GenerationProgress({ message, onImageClick }: GenerationProgressProps) {
  const {
    currentStep = 0,
    totalSteps = 0,
    currentImage,
    isComplete,
    error,
    elapsed,
  } = message;

  const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;
  const clickable = isComplete && currentImage;

  return (
    <div className="message message-generation">
      <div className="generation-card">
        <div
          className={`generation-image-wrap ${!isComplete && !error ? "generating" : ""}`}
          onClick={() => clickable && onImageClick(currentImage!)}
          style={{ cursor: clickable ? "pointer" : "default" }}
        >
          {currentImage ? (
            <img
              className="generation-image"
              src={currentImage}
              alt="Generated"
            />
          ) : (
            <div className="generation-placeholder">
              <div className="loading-spinner" />
              <span>Подготовка...</span>
            </div>
          )}
        </div>

        <div className="generation-footer">
          {error ? (
            <div className="generation-error">{error}</div>
          ) : isComplete ? (
            <>
              <div className="generation-done">
                <TypewriterText text="Генерация завершена!" speed={60} />
              </div>
              {elapsed && (
                <div className="generation-elapsed">{elapsed} сек.</div>
              )}
            </>
          ) : (
            <div className="generation-progress">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span className="progress-text">
                {currentStep} / {totalSteps}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
