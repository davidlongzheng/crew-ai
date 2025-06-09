import { Action, ClientMessage, GameState } from "@/lib/types";
import { getPlayerPosition } from "../utils";
import { GameStatus } from "./GameStatus";
import { PlayerHandle } from "./PlayerHandle";
import { CenterArea } from "./CenterArea";
import { PlayerInfoModal } from "./PlayerInfoModal";
import { useState } from "react";

interface GameAreaProps {
  gameState: GameState;
  uid: string;
  roomId: string;
  sendJsonMessage: (message: ClientMessage) => void;
  currentPlayerIdx: number;
  handleMove: (action: Action) => void;
}

export function GameArea({
  gameState,
  uid,
  roomId,
  sendJsonMessage,
  currentPlayerIdx,
  handleMove,
}: GameAreaProps) {
  const isDraftPhase = gameState.engine_state!.phase === "draft";

  const [modalPlayerIdx, setModalPlayerIdx] = useState<number | null>(null);

  const maybeSetModalPlayerIdx = (idx: number | null) => {
    if (
      idx === null ||
      gameState.engine_state!.assigned_tasks[idx].length > 0 ||
      gameState.engine_state!.signals[idx]
    ) {
      setModalPlayerIdx(idx);
    } else {
      setModalPlayerIdx(null);
    }
  };

  return (
    <div
      className={`relative p-6 shadow-lg bg-slate-50/95 backdrop-blur-sm rounded-xl flex-1 min-h-[450px] lg:min-h-0`}
      onClick={() => {
        maybeSetModalPlayerIdx(null);
      }}
    >
      <GameStatus
        phase={gameState.engine_state!.phase}
        trick={gameState.engine_state!.trick}
      />
      {/* Player positions */}
      {gameState.players.map((playerIdx, idx) => {
        const position = getPlayerPosition(playerIdx, currentPlayerIdx);
        const isCurrentTurn = gameState.player_uids[idx] === gameState.cur_uid;

        return (
          <div
            key={idx}
            className={`absolute ${
              position === "bottom"
                ? "bottom-4 left-1/2 -translate-x-1/2"
                : position === "left"
                ? `left-8 top-1/2 -translate-y-1/2 -rotate-90 translate-x-[-50%]`
                : position === "top"
                ? `top-16 left-1/2 -translate-x-1/2`
                : `right-8 top-1/2 -translate-y-1/2 rotate-90 translate-x-[50%]`
            }`}
            onClick={(e) => {
              maybeSetModalPlayerIdx(playerIdx);
              e.stopPropagation();
            }}
          >
            <PlayerHandle
              handle={gameState.handles[idx]}
              isCaptain={gameState.engine_state!.captain === playerIdx}
              phase={gameState.engine_state!.phase}
              isCurrentTurn={isCurrentTurn}
              tasks={gameState.engine_state!.assigned_tasks[playerIdx]}
              signal={gameState.engine_state!.signals[playerIdx]}
              hand={gameState.engine_state!.hands[playerIdx]}
            />
          </div>
        );
      })}
      {modalPlayerIdx !== null && (
        <PlayerInfoModal
          tasks={gameState.engine_state!.assigned_tasks[modalPlayerIdx]}
          signal={gameState.engine_state!.signals[modalPlayerIdx]}
          hand={gameState.engine_state!.hands[modalPlayerIdx]}
        />
      )}
      {/* Center game area */}
      <CenterArea
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
  );
}
