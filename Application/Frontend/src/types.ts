export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface GameRecommendation {
  appid: number;
  name: string;
  developers: string[];
  publishers: string[];
  price: string | null;
  description: string;
  image_url?: string | null;
}

export interface ConversationResponse {
  conversation_id: string;
  status: 'needs_input' | 'complete';
  message: string;
  recommendations: GameRecommendation[];
}
