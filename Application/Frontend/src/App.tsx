import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { safeImageUrl } from './api';
import type { GameRecommendation } from './types';
import { MAX_MESSAGE_LENGTH, messageLength, useConversation } from './useConversation';
import { useHealth } from './useHealth';

function GameCard({ game }: { game: GameRecommendation }) {
  const [imageFailed, setImageFailed] = useState(false);
  const image = safeImageUrl(game.image_url);
  const showImage = Boolean(image && !imageFailed);
  return (
    <article className="game-card" aria-label={game.name}>
      <div className={`game-card-layout${showImage ? ' has-artwork' : ''}`}>
        {showImage && <img className="game-artwork" src={image!} alt={`${game.name} artwork`} loading="lazy" referrerPolicy="no-referrer" onError={() => setImageFailed(true)} />}
        <div className="game-body">
          <div className="game-heading"><h3>{game.name}</h3><span className="game-price">{game.price ?? 'Price unavailable'}</span></div>
          {game.description && <div className="game-explanation"><span className="explanation-label">Why it fits</span><p className="game-description">{game.description}</p></div>}
          <div className="game-bottom">
            <span className="game-developer">{game.developers.join(', ') || 'Developer unavailable'}</span>
            <details className="game-details">
              <summary>Game details<span className="sr-only"> for {game.name}</span></summary>
              <dl><div><dt>Publishers</dt><dd>{game.publishers.join(', ') || 'Unavailable'}</dd></div><div><dt>Steam AppID</dt><dd>{game.appid}</dd></div></dl>
            </details>
          </div>
        </div>
      </div>
    </article>
  );
}

export default function App() {
  const conversation = useConversation();
  const health = useHealth();
  const input = useRef<HTMLTextAreaElement>(null);
  const results = useRef<HTMLHeadingElement>(null);
  const firstRender = useRef(true);
  const { response, stage, pending, error } = conversation;
  const needsRestart = error?.recovery === 'restart';
  const originalRequest = conversation.messages.find((message) => message.role === 'user')?.content
    || (stage === 0 ? conversation.pendingText || (error ? conversation.draft.trim() : '') : '');
  const length = messageLength(conversation.draft);

  useEffect(() => { if (response) health.markReady(); }, [response, health.markReady]);
  useEffect(() => {
    if (firstRender.current) { firstRender.current = false; return; }
    if (stage === 2) results.current?.focus();
    else input.current?.focus();
  }, [stage, response?.message, conversation.resetCount]);

  function onSubmit(event: FormEvent) { event.preventDefault(); void conversation.submit(); }

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <div className="app-shell">
        <header className="site-header">
          <div className="brand-group"><span className="wordmark">Games Recommender</span><span className="brand-caption">Find your next game</span></div>
          <button className="button button-primary" onClick={() => conversation.reset()}>New search</button>
        </header>
        <main id="main-content" className="main-content">
          <nav className="progress-nav" aria-label="Discovery progress">
            <ol>{['Your preferences', 'Refine preferences', 'Your games'].map((label, index) => (
              <li key={label} aria-current={stage === index ? 'step' : undefined}><span className="step-number" aria-hidden="true">{index + 1}</span><span>{label}</span>{index < stage && <span className="sr-only"> completed</span>}</li>
            ))}</ol>
          </nav>
          {health.status === 'unavailable' && !error && <div className="connection-notice" role="status"><p>We couldn’t reach the API. Check that the backend is running.</p><button className="button button-secondary" onClick={health.recheck}>Retry connection</button></div>}
          <div className="discovery-layout">
            <aside className="discovery-intro" aria-label="Your preferences summary">
              <p className="eyebrow">Game discovery</p>
              <h1>{stage === 2 ? 'Your next game.' : 'Find your next game.'}</h1>
              <p className="intro-description">{stage === 2 ? 'A few games, picked around you.' : 'A little about what you enjoy. A new world to explore.'}</p>
              {originalRequest && <dl className="preferences-summary"><div><dt>You’re looking for</dt><dd>{originalRequest}</dd></div></dl>}
              {conversation.messages.length > 0 && <details className="conversation-history">
                <summary>Conversation so far</summary>
                <ol>{conversation.messages.map((message, index) => <li key={index}><span className="history-role">{message.role === 'user' ? 'You' : 'Games Recommender'}</span><p>{message.content}</p></li>)}</ol>
              </details>}
            </aside>
            <section className="discovery-content" aria-label={stage === 2 ? 'Game recommendations' : 'Your conversation'}>
              {stage === 2 && response ? <>
                <div className="results-heading"><h2 ref={results} tabIndex={-1}>{response.recommendations.length ? 'Games for you' : 'Your search'}</h2>{response.recommendations.length > 0 && <span>{response.recommendations.length} {response.recommendations.length === 1 ? 'pick' : 'picks'}</span>}</div>
                <p className="response-message">{response.message}</p>
                <div className="game-list">{response.recommendations.map((game, index) => <GameCard key={`${game.appid}-${game.image_url ?? ''}-${index}`} game={game} />)}</div>
                {response.recommendations.length === 0 && <div className="empty-results"><p>Try a fresh search with a game theme or feature you enjoy.</p><button className="button button-secondary" onClick={() => conversation.reset()}>Start a new search</button></div>}
              </> : <div className="question-panel">
                <p className="panel-label">{stage === 0 ? 'Start with a feeling' : 'A little more about you'}</p>
                <h2>{stage === 0 ? 'What would you like to play?' : response?.message}</h2>
                <p className="question-description">{stage === 0 ? 'A quiet escape, a strange new world, or something like your favorite game. What sounds good?' : 'Tell us what matters to you. We’ll take it from there.'}</p>
                <form onSubmit={onSubmit} noValidate>
                  <label className="input-label" htmlFor="message">{stage === 0 ? 'Your preferences' : 'Your answer'}</label>
                  <textarea ref={input} id="message" name="message" rows={4} value={conversation.draft} disabled={pending} readOnly={needsRestart}
                    onChange={(event) => conversation.changeDraft(event.target.value)} autoComplete="off"
                    aria-invalid={Boolean(conversation.validation)} aria-describedby={`message-hint${conversation.validation ? ' message-validation' : ''}`}
                    placeholder={stage === 0 ? 'I’d love an atmospheric exploration game…' : 'Here’s what I have in mind…'} />
                  <div className="input-meta"><span id="message-hint">{conversation.restored ? 'Saved preferences. Review and edit before continuing.' : 'A few words is a good start.'}</span><span className={length > MAX_MESSAGE_LENGTH ? 'character-count over-limit' : 'character-count'}><span className="sr-only">Characters used: </span>{length.toLocaleString()} / 4,000</span></div>
                  {conversation.validation && <p id="message-validation" className="inline-validation" role="alert">{conversation.validation}</p>}
                  {error && <div className="request-error" role="alert"><p>{error.message}</p></div>}
                  <div className="form-actions">
                    <p className="pending-message" role="status" aria-live="polite">{pending ? (conversation.longWait ? 'Still working. Some searches take a little longer.' : 'Understanding your preferences…') : ''}</p>
                    {needsRestart ? <button key="restart" type="button" className="button button-primary restart-button" onClick={(event) => { event.preventDefault(); conversation.restartWithPreferences(); }}>Restart with preferences</button>
                      : <button key="continue" className="button button-primary continue-button" type="submit" disabled={pending || !conversation.draft.trim() || length > MAX_MESSAGE_LENGTH}>{pending ? 'Working…' : 'Continue'}</button>}
                  </div>
                </form>
              </div>}
            </section>
          </div>
        </main>
        <footer className="site-footer"><span>Made for your next discovery.</span><span>Steam game recommendations</span></footer>
      </div>
    </>
  );
}
