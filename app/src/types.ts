export interface Settings {
  steps: number;
  guidance: number;
  width: number;
  height: number;
  negative_prompt: string;
  seed: string;
}

export interface UserMessage {
  id: number;
  type: "user";
  text: string;
}

export interface GenerationMessage {
  id: number;
  type: "generation";
  prompt: string;
  currentStep: number;
  totalSteps: number;
  currentImage: string | null;
  isComplete: boolean;
  error: string | null;
  elapsed: number | null;
}

export type Message = UserMessage | GenerationMessage;

export type BridgeStatus = "loading" | "ready" | "error";

export interface BridgeEvent {
  type: string;
  message?: string;
  progress?: number;
  total_steps?: number;
  step?: number;
  total?: number;
  image?: string;
  imageData?: string;
  elapsed?: number;
}

export interface GenerateParams {
  prompt: string;
  negative_prompt: string;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  seed?: number;
}

export interface BridgeAPI {
  generate: (params: GenerateParams) => Promise<void>;
  ping: () => Promise<void>;
  onEvent: (callback: (event: BridgeEvent) => void) => () => void;
}

declare global {
  interface Window {
    bridge: BridgeAPI;
  }
}
