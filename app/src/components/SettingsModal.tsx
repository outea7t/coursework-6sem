import { useState, useEffect, useCallback } from "react";
import type { Settings } from "../types";

interface SettingsModalProps {
  settings: Settings;
  onSave: (settings: Settings) => void;
  onClose: () => void;
}

interface FieldLimits {
  min: number;
  max: number;
  step?: number;
}

const LIMITS: Record<string, FieldLimits> = {
  steps: { min: 1, max: 150, step: 1 },
  guidance: { min: 1, max: 30, step: 0.5 },
  width: { min: 256, max: 2048 },
  height: { min: 256, max: 2048 },
  seed: { min: 0, max: 4294967295 },
};

type ValidationErrors = Partial<Record<string, string>>;

function validate(settings: Record<string, unknown>): ValidationErrors {
  const errors: ValidationErrors = {};

  const steps = Number(settings.steps);
  if (!Number.isInteger(steps) || steps < LIMITS.steps.min || steps > LIMITS.steps.max) {
    errors.steps = `Целое число от ${LIMITS.steps.min} до ${LIMITS.steps.max}`;
  }

  const g = Number(settings.guidance);
  if (isNaN(g) || g < LIMITS.guidance.min || g > LIMITS.guidance.max) {
    errors.guidance = `От ${LIMITS.guidance.min} до ${LIMITS.guidance.max}`;
  }

  for (const dim of ["width", "height"] as const) {
    const v = Number(settings[dim]);
    if (
      !Number.isInteger(v) ||
      v < LIMITS[dim].min ||
      v > LIMITS[dim].max
    ) {
      errors[dim] = `${LIMITS[dim].min}–${LIMITS[dim].max}`;
    } else if (v % 8 !== 0) {
      errors[dim] = "Должно быть кратно 8";
    }
  }

  if (settings.seed !== "" && settings.seed !== undefined) {
    const s = Number(settings.seed);
    if (!Number.isInteger(s) || s < LIMITS.seed.min || s > LIMITS.seed.max) {
      errors.seed = `0–${LIMITS.seed.max} или пусто (случайный)`;
    }
  }

  return errors;
}

export default function SettingsModal({ settings, onSave, onClose }: SettingsModalProps) {
  const [draft, setDraft] = useState<Record<string, unknown>>({ ...settings });
  const [errors, setErrors] = useState<ValidationErrors>({});

  const update = useCallback((key: string, value: unknown) => {
    setDraft((prev) => {
      const next = { ...prev, [key]: value };
      setErrors(validate(next));
      return next;
    });
  }, []);

  useEffect(() => {
    const handleKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const handleSave = () => {
    const errs = validate(draft);
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;
    onSave({
      steps: Number(draft.steps),
      guidance: Number(draft.guidance),
      width: Number(draft.width),
      height: Number(draft.height),
      negative_prompt: String(draft.negative_prompt ?? ""),
      seed: draft.seed === "" ? "" : String(Number(draft.seed)),
    });
  };

  const hasErrors = Object.keys(errors).length > 0;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">Настройки генерации</h2>
          <button className="modal-close" onClick={onClose}>
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="modal-body">
          {/* Steps */}
          <div className="field">
            <label className="field-label">Количество шагов</label>
            <div className="range-wrap">
              <input
                type="range"
                className="range-input"
                min={LIMITS.steps.min}
                max={LIMITS.steps.max}
                step={LIMITS.steps.step}
                value={Number(draft.steps)}
                onChange={(e) => update("steps", Number(e.target.value))}
              />
              <span className="range-value">{String(draft.steps)}</span>
            </div>
            {errors.steps && <span className="field-error">{errors.steps}</span>}
            <span className="field-hint">Рекомендуется 20–30</span>
          </div>

          {/* Guidance */}
          <div className="field">
            <label className="field-label">Guidance Scale (CFG)</label>
            <div className="range-wrap">
              <input
                type="range"
                className="range-input"
                min={LIMITS.guidance.min}
                max={LIMITS.guidance.max}
                step={LIMITS.guidance.step}
                value={Number(draft.guidance)}
                onChange={(e) => update("guidance", Number(e.target.value))}
              />
              <span className="range-value">{String(draft.guidance)}</span>
            </div>
            {errors.guidance && (
              <span className="field-error">{errors.guidance}</span>
            )}
            <span className="field-hint">
              Выше — точнее следует промпту. Рекомендуется 7–12
            </span>
          </div>

          {/* Width & Height */}
          <div className="field">
            <label className="field-label">Размер изображения</label>
            <div className="field-row">
              <div>
                <input
                  type="number"
                  className={`field-input ${errors.width ? "error" : ""}`}
                  value={String(draft.width)}
                  onChange={(e) => update("width", e.target.value)}
                  placeholder="Ширина"
                  min={LIMITS.width.min}
                  max={LIMITS.width.max}
                  step={8}
                />
                {errors.width && (
                  <span className="field-error">{errors.width}</span>
                )}
              </div>
              <div>
                <input
                  type="number"
                  className={`field-input ${errors.height ? "error" : ""}`}
                  value={String(draft.height)}
                  onChange={(e) => update("height", e.target.value)}
                  placeholder="Высота"
                  min={LIMITS.height.min}
                  max={LIMITS.height.max}
                  step={8}
                />
                {errors.height && (
                  <span className="field-error">{errors.height}</span>
                )}
              </div>
            </div>
            <span className="field-hint">Кратно 8. Рекомендуется 1024×1024</span>
          </div>

          {/* Seed */}
          <div className="field">
            <label className="field-label">Seed</label>
            <input
              type="number"
              className={`field-input ${errors.seed ? "error" : ""}`}
              value={String(draft.seed ?? "")}
              onChange={(e) => update("seed", e.target.value)}
              placeholder="Случайный"
              min={LIMITS.seed.min}
              max={LIMITS.seed.max}
            />
            {errors.seed && (
              <span className="field-error">{errors.seed}</span>
            )}
            <span className="field-hint">
              Оставьте пустым для случайного
            </span>
          </div>

          {/* Negative prompt */}
          <div className="field">
            <label className="field-label">Отрицательный промпт</label>
            <textarea
              className="field-input"
              value={String(draft.negative_prompt ?? "")}
              onChange={(e) => update("negative_prompt", e.target.value)}
              placeholder="Что исключить из генерации..."
              rows={2}
            />
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-ghost" onClick={onClose}>
            Отмена
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={hasErrors}
          >
            Сохранить
          </button>
        </div>
      </div>
    </div>
  );
}
