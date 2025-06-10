// Types matching backend schemas
export interface Action {
  player: number;
  type: "draft" | "nodraft" | "signal" | "nosignal" | "play";
  card?: Card;
  task_idx?: number;
  toString(): string;
}

export interface Task {
  formula: string;
  desc: string;
  difficulty: number;
  task_idx: number;
}

export function action_to_string(action: Action, tasks: Record<number, Task>, signals: (Signal | null)[]): string {
  switch (action.type) {
    case "draft":
      return `Drafted "${tasks[action.task_idx!].desc}"`;
    case "nodraft":
      return "Passed";
    case "signal":
      return `Signaled ${card_to_string(action.card!)} as ${signals[action.player]!.value}`;
    case "nosignal":
      return "Passed";
    case "play":
      return `Played ${card_to_string(action.card!)}`;
    default:
      return "Unknown Action";
  }
}

export interface Event {
  type: "action" | "trick_winner" | "new_trick" | "game_ended";
  action?: Action;
  phase?: "draft" | "play" | "signal" | "end";
  trick?: number;
  trick_winner?: number;
}

export interface EngineState {
  phase: string;
  num_players: number;
  hands: Card[][];
  actions: Action[];
  history: Event[];
  trick: number;
  leader: number;
  captain: number;
  cur_player: number;
  active_cards: [Card, number][];
  past_tricks: [Card[], number][];
  signals: (Signal | null)[];
  trick_winner: number | null;
  task_idxs: number[];
  unassigned_task_idxs: number[];
  assigned_tasks: AssignedTask[][];
  status: "success" | "fail" | "unresolved";
  value: number;
}

export interface GameState {
  seqnum: number;
  stage: "lobby" | "play";
  player_uids: string[];
  handles: string[];
  players: number[];
  cur_uid: string | null;
  engine_state: EngineState | null;
  valid_actions: Action[] | null;
  active_cards: [Card, number][];
  tasks: Record<number, Task> | null;
  num_players: number;
  difficulty: number;
}

export interface ServerMessage {
  type: string;
  start_seqnum?: number;
  seqnum?: number;
  uid?: string;
  handle?: string;
  stage?: "lobby" | "play";
  player_uids?: string[];
  handles?: string[];
  players?: number[];
  cur_uid?: string | null;
  engine_state?: EngineState;
  valid_actions?: Action[] | null;
  active_cards?: [Card, number][];
  tasks?: Record<number, Task> | null;
  num_players?: number;
  difficulty?: number;
  trick_winner?: number;
  trick?: number;
  message?: string;
}

export interface ClientMessage {
  type: string;
  room_id: string;
  handle?: string;
  action?: Action;
  num_players?: number;
  difficulty?: number;
}

export interface AssignedTask {
  formula: string;
  desc: string;
  difficulty: number;
  task_idx: number;
  player: number;
  status: "success" | "fail" | "unresolved";
  value: number;
}

export interface Signal {
  card: Card;
  value: "singleton" | "highest" | "lowest" | "other";
  trick: number;
}

export interface Card {
  rank: number;
  suit: number;
  is_trump: boolean;
}

export const TO_SUIT_LETTER: Record<number, string> = {
  0: "b",
  1: "g",
  2: "p",
  3: "y",
  4: "t",
};

export const TO_SUIT_NUM: Record<string, number> = {
  "b": 0,
  "g": 1,
  "p": 2,
  "y": 3,
  "t": 4,
};


export function card_to_string(card: Card): string {
  return `${card.rank}${TO_SUIT_LETTER[card.suit]}`;
}