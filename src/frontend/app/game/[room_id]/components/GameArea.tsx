import { Action, ClientMessage, GameState } from "@/lib/types";
import { getPlayerPosition } from "../utils";
import { GameStatus } from "./GameStatus";
import { PlayerInfo } from "./PlayerInfo";
import { CenterArea } from "./CenterArea";

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

  return (
    <div
      className={`relative p-6 shadow-lg bg-slate-50/95 backdrop-blur-sm rounded-xl ${
        isDraftPhase ? "flex-none min-h-fit" : "flex-1 min-h-96 lg:min-h-0"
      }`}
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
                ? `left-4 top-20`
                : position === "top"
                ? `top-20 left-1/2 -translate-x-1/2`
                : `right-4 top-20`
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
      <div className="w-full px-4 pt-28 pb-20 min-h-[300px]">
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
    </div>
  );
}
