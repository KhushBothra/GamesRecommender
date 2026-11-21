import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import App from './App';
import { first, final } from './test/fixtures';

const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status });

function mockApi(turns: Array<unknown | Error | Promise<Response>>) {
  let index = 0;
  const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (url) => {
    if (String(url).endsWith('/health')) return json({ status: 'ok' });
    const value = turns[index++];
    if (value instanceof Error) throw value;
    return value instanceof Promise ? await value : value instanceof Response ? value : json(value);
  });
  vi.stubGlobal('fetch', fetchMock);
  const calls = () => fetchMock.mock.calls.filter((call) => call[1]?.method === 'POST');
  return { fetchMock, calls };
}

async function answer(user: ReturnType<typeof userEvent.setup>, value: string) {
  await user.type(screen.getByRole('textbox'), value);
  await user.click(screen.getByRole('button', { name: 'Continue' }));
}

describe('Guided Games Recommender', () => {
  it('routes later answers by conversation ID and sends only the latest user_text', async () => {
    const api = mockApi([first, final]);
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByRole('textbox')).toHaveValue('');
    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled();
    await answer(user, 'An exploration game.');
    await screen.findByRole('heading', { name: first.message });
    expect(screen.getByRole('textbox')).toHaveFocus();
    await answer(user, 'Single-player.');
    await screen.findByRole('heading', { name: 'Portal 2' });
    expect(api.calls().map((call) => call[0])).toEqual(['/api/v1/conversations', '/api/v1/conversations/session-1/turns']);
    expect(api.calls().map((call) => JSON.parse(String(call[1]?.body)))).toEqual([{ user_text: 'An exploration game.' }, { user_text: 'Single-player.' }]);
    expect(screen.getByText('0.0')).toBeInTheDocument();
    expect(screen.queryByText('Free')).not.toBeInTheDocument();
    expect(screen.getByText('Price unavailable')).toBeInTheDocument();
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Games for you' })).toHaveFocus();
  });

  it('supports multiple follow-ups and exposes the previous question/answer history', async () => {
    const second = { ...first, message: 'Do you enjoy puzzles?' };
    const api = mockApi([first, second, final]);
    const user = userEvent.setup();
    render(<App />);
    await answer(user, 'Explore a new world.');
    await screen.findByRole('heading', { name: first.message });
    await answer(user, 'Single-player.');
    await screen.findByRole('heading', { name: second.message });
    await user.click(screen.getByText('Conversation so far'));
    expect(screen.getByText('Single-player.')).toBeVisible();
    expect(screen.getByText(first.message)).toBeVisible();
    await answer(user, 'Yes, puzzles.');
    await screen.findByRole('heading', { name: 'Games for you' });
    expect(api.calls()).toHaveLength(3);
  });

  it.each([404, 502])('recovers from HTTP %i with a new editable search rather than replaying the failed turn', async (status) => {
    const api = mockApi([first, json({ detail: 'Failed' }, status), { ...first, conversation_id: 'new-session' }]);
    const user = userEvent.setup();
    render(<App />);
    await answer(user, 'An exploration game.');
    await screen.findByRole('heading', { name: first.message });
    await answer(user, 'Single-player.');
    await screen.findByRole('alert');
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveValue('Single-player.');
    await user.click(screen.getByRole('button', { name: 'Restart with preferences' }));
    expect(screen.getByRole('textbox')).toHaveValue('An exploration game.\n\nAdditional preferences:\nSingle-player.');
    expect(api.calls()).toHaveLength(2);
    await user.click(screen.getByRole('button', { name: 'Continue' }));
    await screen.findByRole('heading', { name: first.message });
    expect(api.calls()[2][0]).toBe('/api/v1/conversations');
  });

  it('lets a 422 answer be corrected in the same conversation', async () => {
    const api = mockApi([first, json({}, 422), final]);
    const user = userEvent.setup();
    render(<App />);
    await answer(user, 'An exploration game.');
    await screen.findByRole('heading', { name: first.message });
    await answer(user, 'Old answer.');
    await screen.findByRole('alert');
    await user.clear(screen.getByRole('textbox'));
    await answer(user, 'A corrected answer.');
    await screen.findByRole('heading', { name: 'Portal 2' });
    expect(api.calls()[2][0]).toBe('/api/v1/conversations/session-1/turns');
    expect(JSON.parse(String(api.calls()[2][1]?.body))).toEqual({ user_text: 'A corrected answer.' });
  });

  it('ignores late responses after New search and prevents duplicate sends', async () => {
    let resolve!: (response: Response) => void;
    const api = mockApi([new Promise<Response>((done) => { resolve = done; }), first]);
    const user = userEvent.setup();
    render(<App />);
    await answer(user, 'Old search.');
    expect(screen.getByRole('textbox')).toBeDisabled();
    fireEvent.submit(screen.getByRole('textbox').closest('form')!);
    expect(api.calls()).toHaveLength(1);
    await user.click(screen.getByRole('button', { name: 'New search' }));
    await answer(user, 'New search.');
    await screen.findByRole('heading', { name: first.message });
    await act(async () => resolve(json(final)));
    expect(screen.queryByRole('heading', { name: 'Games for you' })).not.toBeInTheDocument();
  });

  it('accepts immediate completion and omits broken artwork', async () => {
    mockApi([final]);
    const user = userEvent.setup();
    render(<App />);
    await answer(user, 'I already know my preferences.');
    await screen.findByRole('heading', { name: 'Portal 2' });
    const image = screen.getByRole('img', { name: 'Portal 2 artwork' });
    fireEvent.error(image);
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.getByText('A generated explanation about puzzles.')).toBeInTheDocument();
  });

  it('preserves the backend message for a terminal empty/refused response', async () => {
    mockApi([{ ...final, message: 'I could not map your request to supported descriptors.', recommendations: [] }]);
    const user = userEvent.setup();
    render(<App />);
    await answer(user, 'Something very unusual.');
    await screen.findByRole('heading', { name: 'Your search' });
    expect(screen.getByText('I could not map your request to supported descriptors.', { selector: '.response-message' })).toBeVisible();
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  });

  it('requires editing when recovered preferences exceed 4,000 characters', async () => {
    mockApi([first, json({}, 502)]);
    render(<App />);
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'a'.repeat(3000) } });
    fireEvent.submit(input.closest('form')!);
    await screen.findByRole('heading', { name: first.message });
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'b'.repeat(2000) } });
    fireEvent.submit(screen.getByRole('textbox').closest('form')!);
    await screen.findByRole('alert');
    fireEvent.click(screen.getByRole('button', { name: 'Restart with preferences' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Edit them before continuing');
    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled();
    expect(String((screen.getByRole('textbox') as HTMLTextAreaElement).value)).toContain('b'.repeat(2000));
  });

  it('rejects blank and overlong drafts without posting to the API', () => {
    const api = mockApi([]);
    render(<App />);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '   ' } });
    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled();
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'a'.repeat(4001) } });
    expect(screen.getByRole('alert')).toHaveTextContent('4,000');
    fireEvent.submit(screen.getByRole('textbox').closest('form')!);
    expect(api.calls()).toHaveLength(0);
  });

  it('reassures the user after ten seconds without inventing progress', async () => {
    vi.useFakeTimers();
    let resolve!: (response: Response) => void;
    mockApi([new Promise<Response>((done) => { resolve = done; })]);
    render(<App />);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'An exploration game.' } });
    fireEvent.submit(screen.getByRole('textbox').closest('form')!);
    expect(screen.getByText('Understanding your preferences…')).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(screen.getByText('Still working. Some searches take a little longer.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'New search' }));
    await act(async () => resolve(json(final)));
    expect(screen.getByRole('textbox')).toHaveValue('');
    expect(screen.queryByText('Still working. Some searches take a little longer.')).not.toBeInTheDocument();
  });
});
