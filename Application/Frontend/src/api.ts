import type { ConversationResponse, GameRecommendation } from './types';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL?.trim() || '/api/v1').replace(/\/$/, '');
const configuredTimeout = Number(import.meta.env.VITE_API_TIMEOUT_MS);
export const REQUEST_TIMEOUT = Number.isFinite(configuredTimeout) && configuredTimeout > 0 ? configuredTimeout : 300_000;

export class ApiError extends Error {
  constructor(message: string, public readonly recovery: 'edit' | 'restart' = 'restart') {
    super(message);
    this.name = 'ApiError';
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

export function safeImageUrl(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' ? url.href : null;
  } catch { return null; }
}

function isGame(value: unknown): value is GameRecommendation {
  return isRecord(value)
    && typeof value.appid === 'number' && Number.isInteger(value.appid) && value.appid > 0
    && typeof value.name === 'string'
    && isStringArray(value.developers) && isStringArray(value.publishers)
    && (value.price === null || typeof value.price === 'string')
    && typeof value.description === 'string'
    && (value.image_url === undefined || value.image_url === null || typeof value.image_url === 'string');
}

function isConversationResponse(value: unknown): value is ConversationResponse {
  return isRecord(value)
    && typeof value.conversation_id === 'string' && value.conversation_id.trim().length > 0
    && (value.status === 'needs_input' || value.status === 'complete')
    && typeof value.message === 'string'
    && Array.isArray(value.recommendations) && value.recommendations.every(isGame);
}

async function request(path: string, signal: AbortSignal, options: RequestInit = {}, timeout = REQUEST_TIMEOUT): Promise<unknown> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  let timedOut = false;
  if (signal.aborted) controller.abort();
  signal.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
  try {
    const response = await fetch(`${BASE_URL}${path}`, { ...options, signal: controller.signal });
    if (!response.ok) {
      if (response.status === 422) throw new ApiError('The service could not accept that answer. Edit it and try again.', 'edit');
      if (response.status === 404) throw new ApiError('This conversation has expired or is no longer available. Your preferences are still here.');
      throw new ApiError('We couldn’t complete this search. Your preferences are still here, ready for a fresh start.');
    }
    let value: unknown;
    try { value = await response.json(); }
    catch (error) {
      if (controller.signal.aborted) throw error;
      throw new ApiError('The service returned an unexpected response. Start a fresh search with your saved preferences.');
    }
    return value;
  } catch (error) {
    if (signal.aborted) throw error;
    if (timedOut) throw new ApiError('This request took too long. The server may still be working, so start a fresh search with your saved preferences.');
    if (error instanceof ApiError) throw error;
    throw new ApiError('We lost the connection to the recommendation service. Your preferences are still here.');
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', cancel);
  }
}

export async function sendUserText(userText: string, conversationId: string | null, signal: AbortSignal): Promise<ConversationResponse> {
  const path = conversationId ? `/conversations/${encodeURIComponent(conversationId)}/turns` : '/conversations';
  const response = await request(path, signal, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_text: userText }),
  });
  if (!isConversationResponse(response) || (conversationId && response.conversation_id !== conversationId)) {
    throw new ApiError('The service returned an unexpected response. Start a fresh search with your saved preferences.');
  }
  return response;
}

export async function checkHealth(signal: AbortSignal): Promise<void> {
  const response = await request('/health', signal, {}, 5_000);
  if (!isRecord(response) || response.status !== 'ok') throw new ApiError('The API is unavailable.');
}
