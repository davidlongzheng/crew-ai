import { Action, Card } from "@/lib/types";
import { SUIT_COLORS } from "../constants";
import { formatCardText, getPlayerPosition } from "../utils";

export interface CenterAreaProps {
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


export const CenterArea = ({
  phase,
  win,
  unassignedTasks,
  tasks,
  validActions,
  activeCards,
  currentPlayerIdx,
  onDraft,
  isCurrentTurn,
}: CenterAreaProps) => {
  if (phase === "draft") {
    return (
      <div className="flex flex-col items-center justify-start h-full gap-4 overflow-y-auto">
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
              className={`bg-white/90 backdrop-blur-sm rounded-xl border-2 p-4 w-full max-w-md shadow-lg transition-all duration-200
                ${
                  isDraftable
                    ? "border-blue-500 hover:border-blue-600 cursor-pointer transform hover:scale-105 active:scale-95 hover:shadow-xl"
                    : "border-gray-300 opacity-50"
                }`}
            >
              <div className="text-sm sm:text-xs font-semibold text-gray-800 leading-relaxed break-words">
                {formatCardText(tasks[taskIdx].desc)}
              </div>
              <div className="mt-2 text-xs sm:text-[10px] font-medium text-gray-600">
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
        <div className="p-8 border shadow-xl bg-white/90 backdrop-blur-sm rounded-xl border-white/30">
          <div
            className={`text-4xl font-bold mb-4 text-center ${
              win ? "text-green-600" : "text-red-600"
            }`}
          >
            {win ? "Victory" : "Defeat"}
          </div>
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
              } rounded-lg border-2 flex items-center justify-center shadow-lg
              ${
                card.is_trump
                  ? "border-yellow-500 shadow-yellow-500/25"
                  : "border-white shadow-lg"
              }`}
          >
            <div className="flex flex-col items-center">
              <span className="text-lg font-bold text-white">{card.rank}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
};
