import { describe, expect, it, vi } from 'vitest';
import { ApiError, REQUEST_TIMEOUT, safeImageUrl, sendUserText } from './api';
import { final } from './test/fixtures';

describe('Stateful API client', () => {
  it.each([404, 502])('marks HTTP %i for restart instead of automatic retry', async (status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status })));
    try { await sendUserText('Answer.', 'session-1', new AbortController().signal); throw new Error('Expected rejection'); }
    catch (error) { expect(error).toBeInstanceOf(ApiError); expect((error as ApiError).recovery).toBe('restart'); }
  });
  it('marks validation failures as editable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 422 })));
    await expect(sendUserText('Answer.', 'id', new AbortController().signal)).rejects.toMatchObject({ recovery: 'edit' });
  });
  it.each(['invalid json', JSON.stringify({ ...final, status: 'unknown' }), JSON.stringify({ ...final, recommendations: [{ name: 'Missing fields' }] })])('rejects malformed responses', async (body) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body)));
    await expect(sendUserText('Answer.', null, new AbortController().signal)).rejects.toThrow(/unexpected response/);
  });
  it('rejects a response belonging to a different conversation', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(final))));
    await expect(sendUserText('Answer.', 'other-session', new AbortController().signal)).rejects.toThrow(/unexpected response/);
  });
  it('allows missing optional image_url from older backend versions', async () => {
    const response = { ...final, recommendations: final.recommendations.map(({ image_url: _image, ...rest }) => rest) };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(response))));
    await expect(sendUserText('Answer.', null, new AbortController().signal)).resolves.toEqual(response);
  });
  it('times out a stalled request with restart recovery', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn().mockImplementation((_url, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    })));
    const assertion = expect(sendUserText('Answer.', null, new AbortController().signal)).rejects.toMatchObject({ recovery: 'restart', message: expect.stringContaining('took too long') });
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT);
    await assertion;
  });
  it('only accepts secure artwork URLs', () => {
    expect(safeImageUrl('javascript:alert(1)')).toBeNull();
    expect(safeImageUrl('http://example.com/a.jpg')).toBeNull();
    expect(safeImageUrl('https://shared.fastly.steamstatic.com/a.jpg')).toBe('https://shared.fastly.steamstatic.com/a.jpg');
  });
});
