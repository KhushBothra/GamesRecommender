export const first = { conversation_id: 'session-1', status: 'needs_input', message: 'Single-player or multiplayer?', recommendations: [] };
export const final = {
  conversation_id: 'session-1', status: 'complete', message: 'Here are two games for you.',
  recommendations: [
    { appid: 620, name: 'Portal 2', developers: ['Valve'], publishers: ['Valve'], price: '0.0', description: 'A generated explanation about puzzles.', image_url: 'https://shared.fastly.steamstatic.com/header.jpg' },
    { appid: 413150, name: 'Stardew Valley', developers: ['ConcernedApe'], publishers: ['ConcernedApe'], price: null, description: 'A generated explanation about exploration.', image_url: null },
  ],
};
