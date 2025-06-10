"use client";

import { Action, ClientMessage, GameState } from "@/lib/types";
import { GameArea } from "./components/GameArea";
import { ActionHistory } from "./components/ActionHistory";
import { PlayerHand } from "./components/PlayerHand";

interface PlayStageProps {
  gameState: GameState;
  uid: string;
  roomId: string;
  sendJsonMessage: (message: ClientMessage) => void;
}

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

  const hasFailedTasks = gameState.engine_state!.assigned_tasks.some((tasks) =>
    tasks.some((task) => task.status === "fail")
  );

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="max-w-7xl mx-auto min-h-[calc(225vh-6rem)] lg:min-h-[calc(100vh-6rem)] lg:h-[calc(100vh-6rem)] relative z-10 mt-4 px-4 lg:px-0">
        {/* Header with game controls */}
        <div className="grid grid-cols-2 mb-3">
          <div className="flex justify-end col-start-3 gap-2">
            <button
              onClick={() => {
                sendJsonMessage({
                  type: "end_game",
                  room_id: roomId,
                });
              }}
              className="px-3 py-3 text-xs font-semibold text-white transition-all duration-200 transform bg-red-600 border border-gray-400 rounded-lg shadow-lg hover:bg-red-700 hover:scale-105 active:scale-95 hover:shadow-xl sm:min-w-[100px]"
            >
              End Game
            </button>
            {(gameState.engine_state!.phase === "end" || hasFailedTasks) && (
              <button
                onClick={() => {
                  sendJsonMessage({
                    type: "start_game",
                    room_id: roomId,
                  });
                }}
                className="px-3 py-3 text-xs font-semibold text-white transition-all duration-200 transform bg-green-600 border border-gray-400 rounded-lg shadow-lg hover:bg-green-700 hover:scale-105 active:scale-95 hover:shadow-xl min-w-[100px]"
              >
                Play Again
              </button>
            )}
          </div>
        </div>
        <div
          className={`grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-4 min-h-full lg:h-full`}
        >
          <div className="flex flex-col gap-4">
            <GameArea
              gameState={gameState}
              uid={uid}
              currentPlayerIdx={currentPlayerIdx}
              handleMove={handleMove}
            />

            <PlayerHand
              phase={gameState.engine_state!.phase}
              isCurrentTurn={gameState.cur_uid === uid}
              validActions={gameState.valid_actions}
              hand={gameState.engine_state!.hands[currentPlayerIdx]}
              onMove={handleMove}
            />
          </div>
          <div className="flex flex-col p-6 overflow-hidden border shadow-lg lg:flex bg-white/80 backdrop-blur-sm rounded-xl border-white/30 max-h-[500px] lg:max-h-none">
            <h2 className="mb-6 text-xl font-semibold text-gray-800">
              History
            </h2>
            <ActionHistory
              history={gameState.engine_state!.history}
              handles={gameState.handles}
              players={gameState.players}
              tasks={gameState.tasks!}
              signals={gameState.engine_state!.signals}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
