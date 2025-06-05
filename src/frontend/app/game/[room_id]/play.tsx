"use client";

import {
  Action,
  ClientMessage,
  GameState,
  Signal,
  Card,
  action_to_string,
  card_to_string,
  TO_SUIT_NUM,
} from "@/lib/types";

interface PlayStageProps {
  gameState: GameState;
  uid: string;
  roomId: string;
  sendJsonMessage: (message: ClientMessage) => void;
}

// Constants
const SUIT_COLORS = {
  0: "bg-[#3498db]", // blue
  1: "bg-[#2ecc71]", // green
  2: "bg-[#ff69b4]", // pink
  3: "bg-[#f1c40f]", // yellow
  4: "bg-[#1a2530]", // trump
} as const;

const PHASE_COLORS = {
  draft: "bg-[#2ecc71]", // green
  signal: "bg-[#f1c40f]", // yellow
  play: "bg-[#3498db]", // blue
} as const;

// Types
type PlayerPosition = "top" | "bottom" | "left" | "right";

interface GameStatusProps {
  phase: string;
  trick: number;
}

interface PlayerInfoProps {
  handle: string;
  phase: string;
  isCaptain: boolean;
  isLeader: boolean;
  isCurrentTurn: boolean;
  tasks: Array<{ desc: string; status: string }>;
  signal?: Signal | null;
  hand: Card[];
}

interface CardProps {
  card: Card;
  isPlayable?: boolean;
  onClick?: () => void;
  isAnimating?: boolean;
}

interface GameAreaProps {
  phase: string;
  win: boolean;
  unassignedTasks: number[];
  tasks: Record<number, { desc: string; difficulty: number }>;
  validActions: Action[] | null;
  activeCards: [Card, number][];
  currentPlayerIdx: number;
  onDraft: (action: Action) => void;
  isCurrentTurn: boolean;
}

interface ActionHistoryProps {
  history: Array<{
    type: string;
    action?: Action;
    phase?: string;
    trick?: number;
    trick_winner?: number;
  }>;
  handles: string[];
  players: number[];
  tasks: Record<
    number,
    { desc: string; difficulty: number; formula: string; task_idx: number }
  >;
}

interface PlayerHandProps {
  phase: string;
  isCurrentTurn: boolean;
  validActions: Action[] | null;
  hand: Array<{ rank: number; suit: number; is_trump: boolean }>;
  onMove: (action: Action) => void;
}

const formatCardText = (text: string) => {
  const cardRegex = /(\d[bgypt])/g;
  const parts = text.split(cardRegex);

  return parts.map((part, idx) => {
    if (cardRegex.test(part)) {
      const rank = part[0];
      const suit = TO_SUIT_NUM[part[1]];
      return (
        <span
          key={idx}
          className={`inline-flex items-center -translate-y-0.5 ${
            SUIT_COLORS[suit as keyof typeof SUIT_COLORS]
          } rounded-none border border-white px-0.5`}
        >
          <span className="text-[8px] font-['Press_Start_2P'] text-white">
            {rank}
          </span>
        </span>
      );
    }
    return part;
  });
};

// Components
const GameStatus = ({ phase, trick }: GameStatusProps) => (
  <div
    className={`absolute top-4 left-1/2 -translate-x-1/2 ${
      PHASE_COLORS[phase as keyof typeof PHASE_COLORS]
    } px-6 py-2 rounded-none border-2 border-white z-10`}
  >
    <span className="font-['Press_Start_2P'] text-white text-xs">
      {phase === "draft"
        ? `Draft Round ${trick + 1}`
        : phase === "end"
        ? "Game Over"
        : `Trick ${trick + 1} - ${phase == "play" ? "Play" : "Signal"}`}
    </span>
  </div>
);

const PlayerInfo = ({
  handle,
  phase,
  isCaptain,
  isLeader,
  isCurrentTurn,
  tasks,
  signal,
  hand,
}: PlayerInfoProps) => {
  const playerInfoBox = (tasks.length > 0 || signal) && (
    <div className="bg-[#34495e] rounded-none border-2 border-white p-3 w-48">
      <ul className="space-y-1">
        {[
          ...tasks,
          ...(signal
            ? [
                {
                  desc: `${card_to_string(signal.card)} is ${signal.value}.`,
                  status: hand?.some(
                    (card) =>
                      card.rank === signal.card.rank &&
                      card.suit === signal.card.suit
                  )
                    ? "unplayed"
                    : "played",
                },
              ]
            : []),
        ].map((item, idx) => (
          <li
            key={idx}
            className={`text-[10px] font-['Press_Start_2P'] ${
              item.status === "success"
                ? "text-[#2ecc71]"
                : item.status === "fail"
                ? "text-[#e74c3c]"
                : item.status === "unplayed"
                ? "text-[#f1c40f]"
                : item.status === "played"
                ? "text-gray-400"
                : "text-white"
            } border-b border-white/20 last:border-b-0 pb-1`}
          >
            {formatCardText(item.desc)}
          </li>
        ))}
      </ul>
    </div>
  );

  const playerLabel = (
    <div
      className={`px-4 py-2 rounded-none border-2 ${
        isCurrentTurn
          ? `${PHASE_COLORS[phase as keyof typeof PHASE_COLORS]} border-white`
          : "bg-[#34495e] border-white"
      } ${
        isLeader ? "text-[#40e0d0]" : "text-white"
      } font-['Press_Start_2P'] text-sm`}
    >
      {isCaptain && (
        <svg
          className="inline-block w-4 h-4 text-[#f39c12] mr-1 -translate-y-0.5"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z" />
          <path d="M12 6c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 10c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4z" />
        </svg>
      )}
      {handle}
    </div>
  );

  return (
    <div className="flex flex-col items-center gap-4">
      {playerLabel}
      {playerInfoBox}
    </div>
  );
};

const CardBox = ({ card, isPlayable, onClick, isAnimating }: CardProps) => {
  const suitColor = SUIT_COLORS[card.suit as keyof typeof SUIT_COLORS];

  return (
    <div
      onClick={onClick}
      className={`w-16 h-24 ${suitColor} rounded-none border-2 flex items-center justify-center
        ${
          isPlayable
            ? "cursor-pointer transition-all duration-200 hover:shadow-xl hover:-translate-y-1 transform hover:scale-105 active:scale-95"
            : "opacity-50"
        }
        ${
          card.is_trump
            ? "border-[#f1c40f] hover:border-[#f39c12]"
            : "border-white hover:border-[#ecf0f1]"
        }
        ${isAnimating ? "animate-card-play" : ""}`}
    >
      <div className="flex flex-col items-center">
        <span className="text-lg font-['Press_Start_2P'] text-white">
          {card.rank}
        </span>
      </div>
    </div>
  );
};

const GameArea = ({
  phase,
  win,
  unassignedTasks,
  tasks,
  validActions,
  activeCards,
  currentPlayerIdx,
  onDraft,
  isCurrentTurn,
}: GameAreaProps) => {
  if (phase === "draft") {
    return (
      <div className="flex flex-col items-center gap-4 h-full justify-center">
        {unassignedTasks.map((taskIdx) => {
          const isDraftable =
            isCurrentTurn &&
            validActions?.some(
              (action) => action.type === "draft" && action.task_idx === taskIdx
            );

          return (
            <div
              key={taskIdx}
              onClick={() => {
                if (isDraftable) {
                  const draftAction = validActions?.find(
                    (action) =>
                      action.type === "draft" && action.task_idx === taskIdx
                  );
                  if (draftAction) onDraft(draftAction);
                }
              }}
              className={`bg-[#34495e] rounded-none border-2 p-2 max-w-[300px]
                ${
                  isDraftable
                    ? "border-[#3498db] hover:border-[#2980b9] cursor-pointer transform hover:scale-105 active:scale-95"
                    : "border-white opacity-50"
                }`}
            >
              <div className="text-xs font-['Press_Start_2P'] text-white">
                {formatCardText(tasks[taskIdx].desc)}
              </div>
              <div className="text-[10px] font-['Press_Start_2P'] text-[#95a5a6] mt-0.5">
                Difficulty: {tasks[taskIdx].difficulty}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  if (phase === "end") {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div
          className={`text-4xl font-['Press_Start_2P'] mb-4 ${
            win ? "text-[#2ecc71]" : "text-[#e74c3c]"
          }`}
        >
          {win ? "Victory" : "Defeat"}
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full">
      {activeCards.map(([card, playerIdx], idx) => {
        const position = getPlayerPosition(playerIdx, currentPlayerIdx);
        const offset = {
          top:
            position === "top"
              ? "-mt-24"
              : position === "bottom"
              ? "mt-24"
              : "",
          left:
            position === "left"
              ? "-ml-24"
              : position === "right"
              ? "ml-24"
              : "",
        };

        return (
          <div
            key={idx}
            className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-24
              ${offset.top} ${offset.left}
              ${
                SUIT_COLORS[card.suit as keyof typeof SUIT_COLORS]
              } rounded-none border-2 flex items-center justify-center
              ${
                card.is_trump
                  ? "border-[#f1c40f] shadow-[0_0_8px_rgba(241,196,15,0.5)]"
                  : "border-white shadow-[0_0_8px_rgba(255,255,255,0.2)]"
              }`}
          >
            <div className="flex flex-col items-center">
              <span className="text-lg font-['Press_Start_2P'] text-white">
                {card.rank}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
};

const ActionHistory = ({
  history,
  handles,
  players,
  tasks,
}: ActionHistoryProps) => (
  <div
    className="flex-1 overflow-y-auto space-y-2"
    ref={(el) => el?.scrollTo(0, el.scrollHeight)}
  >
    {history.map((event, idx) => (
      <div
        key={idx}
        className="text-xs p-2 rounded-none border-2 border-white bg-[#2c3e50]"
      >
        {event.type === "action" && event.action && (
          <>
            <span className="font-['Press_Start_2P'] text-[#3498db]">
              {handles[players.indexOf(event.action.player)]}:{" "}
            </span>
            <span className="font-['Press_Start_2P'] text-white">
              {formatCardText(action_to_string(event.action, tasks))}
            </span>
          </>
        )}
        {event.type === "trick_winner" && event.trick_winner !== null && (
          <span className="font-['Press_Start_2P'] text-[#2ecc71]">
            Trick {event.trick! + 1} won by{" "}
            {handles[players.indexOf(event.trick_winner!)]}
          </span>
        )}
        {event.type === "new_trick" && (
          <span className="font-['Press_Start_2P'] text-[#9b59b6]">
            Starting {event.phase === "draft" ? "Draft Round " : "Trick "}
            {event.trick! + 1}
          </span>
        )}
        {event.type === "game_ended" && (
          <span className="font-['Press_Start_2P'] text-[#e74c3c]">
            Game Over
          </span>
        )}
      </div>
    ))}
  </div>
);

const PlayerHand = ({
  phase,
  isCurrentTurn,
  validActions,
  hand,
  onMove,
}: PlayerHandProps) => {
  return (
    <div className="bg-[#34495e] rounded-none border-4 border-white p-4">
      {isCurrentTurn && validActions && (
        <div className="text-white font-['Press_Start_2P'] text-[10px] mb-4">
          {phase === "draft"
            ? "It's your turn to draft! Click on one of the tasks above or pass."
            : phase === "signal"
            ? "It's your turn to signal! Click on one of the cards below or pass."
            : "It's your turn to play! Click on one of the cards below."}
        </div>
      )}
      <div className="flex items-center justify-between">
        <div className="flex gap-4">
          {hand.map((card, idx) => {
            const isPlayable =
              isCurrentTurn &&
              validActions?.some(
                (action) =>
                  (action.type === "play" || action.type === "signal") &&
                  action.card?.rank === card.rank &&
                  action.card?.suit === card.suit
              );

            return (
              <CardBox
                key={idx}
                card={card}
                isPlayable={isPlayable}
                onClick={() => {
                  if (isPlayable) {
                    const playAction = validActions?.find(
                      (action) =>
                        (action.type === "play" || action.type === "signal") &&
                        action.card?.rank === card.rank &&
                        action.card?.suit === card.suit
                    );
                    if (playAction) onMove(playAction);
                  }
                }}
              />
            );
          })}
        </div>
        {isCurrentTurn && validActions && (
          <div className="flex gap-2">
            {validActions.some(
              (action) =>
                action.type === "nodraft" || action.type === "nosignal"
            ) && (
              <button
                onClick={() => {
                  const passAction = validActions?.find(
                    (action) =>
                      action.type === "nodraft" || action.type === "nosignal"
                  );
                  if (passAction) onMove(passAction);
                }}
                className="bg-[#95a5a6] hover:bg-[#7f8c8d] text-white px-4 py-2 rounded-none border-2 border-white font-['Press_Start_2P'] text-sm transition-colors transform hover:scale-105 active:scale-95"
              >
                Pass
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Utility functions
const getPlayerPosition = (
  playerIdx: number,
  currentPlayerIdx: number
): PlayerPosition => {
  const relativePos = (playerIdx - currentPlayerIdx + 4) % 4;
  switch (relativePos) {
    case 0:
      return "bottom";
    case 1:
      return "left";
    case 2:
      return "top";
    case 3:
      return "right";
    default:
      return "bottom";
  }
};

// Add to the top of the file, after imports:
const styles = `
@keyframes card-play {
  0% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.5);
  }
  100% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes trick-win {
  0% {
    transform: translate(-50%, -50%) scale(1);
  }
  50% {
    transform: translate(-50%, -50%) scale(1.1);
  }
  100% {
    transform: translate(-50%, -50%) scale(1);
  }
}

.animate-card-play {
  animation: card-play 1s ease-out;
}

.animate-trick-win {
  animation: trick-win 1s ease-out;
}
`;

// Main component
export function PlayStage({
  gameState,
  uid,
  roomId,
  sendJsonMessage,
}: PlayStageProps) {
  const handleMove = (action: Action) => {
    sendJsonMessage({
      type: "move",
      room_id: roomId,
      action: action,
    });
  };

  const currentPlayerIdx = gameState.player_uids.includes(uid)
    ? gameState.players[gameState.player_uids.indexOf(uid)]
    : 0;

  return (
    <div className="min-h-screen bg-[#2c3e50] relative overflow-hidden">
      <style>{styles}</style>
      <div className="absolute inset-0 bg-[url('/pixel-pattern.svg')] opacity-10"></div>

      <div className="max-w-7xl mx-auto h-[calc(100vh-2rem)] grid grid-cols-[1fr_300px] gap-4 relative z-10 mt-4">
        <div className="flex flex-col gap-4">
          {/* Main game area */}
          <div className="relative bg-[#34495e] rounded-none border-4 border-white p-4 flex-1">
            <GameStatus
              phase={gameState.engine_state!.phase}
              trick={gameState.engine_state!.trick}
            />

            {/* Game controls */}
            <div className="absolute top-4 right-4 flex gap-2">
              <button
                onClick={() => {
                  sendJsonMessage({
                    type: "end_game",
                    room_id: roomId,
                  });
                }}
                className="bg-[#e74c3c] hover:bg-[#c0392b] text-white px-4 py-2 rounded-none border-2 border-white font-['Press_Start_2P'] text-sm transition-colors transform hover:scale-105 active:scale-95"
              >
                End Game
              </button>
              {gameState.engine_state!.phase === "end" && (
                <button
                  onClick={() => {
                    sendJsonMessage({
                      type: "start_game",
                      room_id: roomId,
                    });
                  }}
                  className="bg-[#2ecc71] hover:bg-[#27ae60] text-white px-4 py-2 rounded-none border-2 border-white font-['Press_Start_2P'] text-sm transition-colors transform hover:scale-105 active:scale-95"
                >
                  Play Again
                </button>
              )}
            </div>

            {/* Player positions */}
            {gameState.players.map((playerIdx, idx) => {
              const position = getPlayerPosition(playerIdx, currentPlayerIdx);
              const isCurrentTurn =
                gameState.player_uids[idx] === gameState.cur_uid;

              return (
                <div
                  key={idx}
                  className={`absolute ${
                    position === "bottom"
                      ? "bottom-4 left-1/2 -translate-x-1/2"
                      : position === "left"
                      ? "left-4 top-1/2 -translate-y-1/2"
                      : position === "top"
                      ? "top-20 left-1/2 -translate-x-1/2"
                      : "right-4 top-1/2 -translate-y-1/2"
                  }`}
                >
                  <PlayerInfo
                    handle={gameState.handles[idx]}
                    isCaptain={gameState.engine_state!.captain === playerIdx}
                    isLeader={gameState.engine_state!.leader === playerIdx}
                    phase={gameState.engine_state!.phase}
                    isCurrentTurn={isCurrentTurn}
                    tasks={gameState.engine_state!.assigned_tasks[playerIdx]}
                    signal={gameState.engine_state!.signals[playerIdx]}
                    hand={gameState.engine_state!.hands[playerIdx]}
                  />
                </div>
              );
            })}

            {/* Center game area */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2/3 h-[calc(100%-12rem)]">
              <GameArea
                phase={gameState.engine_state!.phase}
                win={gameState.engine_state!.status === "success"}
                unassignedTasks={gameState.engine_state!.unassigned_task_idxs}
                tasks={gameState.tasks!}
                validActions={gameState.valid_actions}
                activeCards={gameState.active_cards}
                currentPlayerIdx={currentPlayerIdx}
                onDraft={handleMove}
                isCurrentTurn={gameState.cur_uid === uid}
              />
            </div>
          </div>

          {/* Player's hand */}
          <PlayerHand
            phase={gameState.engine_state!.phase}
            isCurrentTurn={gameState.cur_uid === uid}
            validActions={gameState.valid_actions}
            hand={gameState.engine_state!.hands[currentPlayerIdx]}
            onMove={handleMove}
          />
        </div>

        {/* Action history sidebar */}
        <div className="bg-[#34495e] rounded-none border-4 border-white p-4 overflow-hidden flex flex-col">
          <h2 className="text-lg font-['Press_Start_2P'] text-white mb-4">
            History
          </h2>
          <ActionHistory
            history={gameState.engine_state!.history}
            handles={gameState.handles}
            players={gameState.players}
            tasks={gameState.tasks!}
          />
        </div>
      </div>
    </div>
  );
}
