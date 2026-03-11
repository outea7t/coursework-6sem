import { useRef, useEffect, useState, type KeyboardEvent, type ChangeEvent } from "react";
import ChatMessage from "./ChatMessage";
import GenerationProgress from "./GenerationProgress";
import type { Message, UserMessage, GenerationMessage } from "../types";

interface ChatProps {
  messages: Message[];
  onSend: (text: string) => void;
  isGenerating: boolean;
  isReady: boolean;
  onImageClick: (src: string) => void;
}

export default function Chat({ messages, onSend, isGenerating, isReady, onImageClick }: ChatProps) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = () => {
    if (!input.trim() || isGenerating || !isReady) return;
    onSend(input);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInput = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="chat">
      <div className="chat-messages">
        {isEmpty ? (
          <div className="empty-state">
            <div className="empty-state-content">
              <div className="empty-state-icon">
                <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <path d="M21 15l-5-5L5 21" />
                </svg>
              </div>
              <p className="empty-state-title">Опишите изображение</p>
              <p className="empty-state-subtitle">
                Напишите промпт для генерации
              </p>
            </div>
          </div>
        ) : (
          <div className="chat-messages-inner">
            {messages.map((msg) =>
              msg.type === "user" ? (
                <ChatMessage key={msg.id} message={msg as UserMessage} />
              ) : (
                <GenerationProgress
                  key={msg.id}
                  message={msg as GenerationMessage}
                  onImageClick={onImageClick}
                />
              ),
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="input-bar">
        <div
          className={`input-container ${!isReady || isGenerating ? "disabled" : ""}`}
        >
          <textarea
            ref={textareaRef}
            className="input-field"
            rows={1}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={
              !isReady
                ? "Загрузка модели..."
                : isGenerating
                  ? "Генерация..."
                  : "Опишите изображение..."
            }
            disabled={!isReady || isGenerating}
          />
          <button
            className="send-btn"
            onClick={handleSubmit}
            disabled={!input.trim() || !isReady || isGenerating}
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="5 12 12 5 19 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
