import { useCallback, useEffect, useRef, useState } from 'react';
import { checkHealth } from './api';

export function useHealth() {
  const [status, setStatus] = useState<'checking' | 'ready' | 'unavailable'>('checking');
  const request = useRef<AbortController | null>(null);
  const recheck = useCallback(() => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setStatus('checking');
    void checkHealth(controller.signal).then(() => {
      if (!controller.signal.aborted) setStatus('ready');
    }).catch(() => {
      if (!controller.signal.aborted) setStatus('unavailable');
    });
  }, []);
  const markReady = useCallback(() => {
    request.current?.abort();
    setStatus('ready');
  }, []);
  useEffect(() => {
    recheck();
    return () => request.current?.abort();
  }, [recheck]);
  return { status, recheck, markReady };
}
