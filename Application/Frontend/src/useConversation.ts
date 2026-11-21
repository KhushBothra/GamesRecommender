import { useEffect, useRef, useState } from 'react';
import { ApiError, sendUserText } from './api';
import type { ChatMessage, ConversationResponse } from './types';

export const MAX_MESSAGE_LENGTH = 4_000;
export const messageLength = (text: string) => Array.from(text).length;

export function useConversation() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [response, setResponse] = useState<ConversationResponse | null>(null);
  const [draft, setDraft] = useState('');
  const [pendingText, setPendingText] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [validation, setValidation] = useState<string | null>(null);
  const [resetCount, setResetCount] = useState(0);
  const [longWait, setLongWait] = useState(false);
  const [restored, setRestored] = useState(false);
  const request = useRef<AbortController | null>(null);
  const generation = useRef(0);

  useEffect(() => () => { generation.current += 1; request.current?.abort(); }, []);
  useEffect(() => {
    if (pendingText === null) { setLongWait(false); return; }
    const timer = setTimeout(() => setLongWait(true), 10_000);
    return () => clearTimeout(timer);
  }, [pendingText]);

  async function submit() {
    if (request.current || response?.status === 'complete' || error?.recovery === 'restart') return;
    const text = draft.trim();
    if (!text) { setValidation('Add a few words about what you’d like to play.'); return; }
    if (messageLength(draft) > MAX_MESSAGE_LENGTH) { setValidation('Keep your message to 4,000 characters or fewer.'); return; }
    const controller = new AbortController();
    const id = ++generation.current;
    request.current = controller;
    setPendingText(text); setError(null); setValidation(null); setRestored(false);
    try {
      const result = await sendUserText(text, conversationId, controller.signal);
      if (generation.current !== id) return;
      setMessages([...messages, { role: 'user', content: text }, { role: 'assistant', content: result.message }]);
      setConversationId(result.conversation_id); setResponse(result); setDraft('');
    } catch (cause) {
      if (generation.current !== id || controller.signal.aborted) return;
      setError(cause instanceof ApiError ? cause : new ApiError('We couldn’t complete this search. Your preferences are still here.'));
    } finally {
      if (generation.current === id) { request.current = null; setPendingText(null); }
    }
  }

  function changeDraft(value: string) {
    setDraft(value);
    if (error?.recovery === 'edit') setError(null);
    setValidation(messageLength(value) > MAX_MESSAGE_LENGTH ? 'Keep your message to 4,000 characters or fewer.' : null);
  }

  function reset(prefill = '') {
    generation.current += 1; request.current?.abort(); request.current = null;
    setMessages([]); setConversationId(null); setResponse(null); setDraft(prefill);
    setPendingText(null); setError(null); setLongWait(false); setRestored(Boolean(prefill));
    setValidation(messageLength(prefill) > MAX_MESSAGE_LENGTH ? 'Your saved preferences are longer than 4,000 characters. Edit them before continuing.' : null);
    setResetCount((count) => count + 1);
  }

  function restartWithPreferences() {
    const saved = messages.filter((message) => message.role === 'user').map((message) => message.content);
    const unsent = draft.trim();
    if (unsent) saved.push(unsent);
    const prefill = saved.length > 1 ? `${saved[0]}\n\nAdditional preferences:\n${saved.slice(1).join('\n\n')}` : saved[0] || '';
    reset(prefill);
  }

  const stage = response?.status === 'complete' ? 2 : response?.status === 'needs_input' ? 1 : 0;
  return {
    messages, response, draft, changeDraft, submit, reset, restartWithPreferences,
    pending: pendingText !== null, pendingText, error, validation, resetCount,
    stage, longWait, restored,
  };
}
