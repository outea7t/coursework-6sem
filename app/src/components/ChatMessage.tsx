import type { UserMessage } from "../types";

interface ChatMessageProps {
  message: UserMessage;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  return (
    <div className="message message-user">
      <div className="bubble">{message.text}</div>
    </div>
  );
}
