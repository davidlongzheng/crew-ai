"use client";

import { Action, ClientMessage, GameState } from "@/lib/types";
import { GameArea } from "./components/GameArea";
import { ActionHistory } from "./components/ActionHistory";
import { PlayerHand } from "./components/PlayerHand";
import { useState } from "react";

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

  const [showHelpModal, setShowHelpModal] = useState(false);

  const currentPlayerIdx = gameState.player_uids.includes(uid)
    ? gameState.players[gameState.player_uids.indexOf(uid)]
    : 0;

  const hasFailedTasks = gameState.engine_state!.assigned_tasks.some((tasks) =>
    tasks.some((task) => task.status === "fail")
  );

  return (
    <div
      className="relative min-h-screen overflow-hidden"
      onClick={() => {
        setShowHelpModal(false);
      }}
    >
      <div className="max-w-7xl mx-auto min-h-[calc(225vh-6rem)] lg:min-h-[calc(100vh-6rem)] lg:h-[calc(100vh-6rem)] relative z-10 mt-4 px-4 lg:px-0">
        {showHelpModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
            <div className="min-w-[40%] w-[60%] p-4 border-2 rounded-lg shadow-xl pointer-events-auto bg-white/95 backdrop-blur-sm border-black min-h-[40%] overflow-y-auto">
              <h2 className="mb-4 text-xl font-bold">How to Play</h2>
              <p className="mb-4 text-sm">
                The Crew is a collaborative trick taking game. The player with
                the trump (black) 4 is the captain, indicated with a C icon.
                Starting from the captain, players draft tasks over three
                rounds. Then players play out their hands in tricks. Players
                only win if all tasks are successful.
              </p>
              <p className="mb-4 text-sm">
                Once per game, players may signal a card from their hand at the
                start of a trick. Players may only signal the highest, lowest,
                or singleton cards of a suit and may not signal trump cards.
              </p>
              <p className="mb-4 text-sm">
                Tasks and signals can be viewed at any time by clicking on the
                player&apos;s handle. Icons are used to indicate the status of
                each task as well as any active signals.
              </p>
            </div>
          </div>
        )}
        {/* Header with game controls */}
        <div className="grid grid-cols-2 mb-3">
          <div className="flex items-center">
            <button
              onClick={(e) => {
                setShowHelpModal(true);
                e.stopPropagation();
              }}
              className="px-3 py-3 text-xs font-semibold text-white transition-all duration-200 transform bg-blue-600 border border-gray-400 rounded-lg shadow-lg hover:bg-blue-700 hover:scale-105 active:scale-95 hover:shadow-xl"
            >
              Help
            </button>
          </div>
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
