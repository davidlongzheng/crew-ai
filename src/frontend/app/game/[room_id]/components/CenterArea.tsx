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
      <div className="absolute flex flex-col items-center justify-start h-[60%] gap-2 lg:gap-4 py-1 overflow-y-auto -translate-x-1/2 top-[110px] left-1/2 w-[80%] sm:min-w-[200px]">
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
              className={`bg-white/90 backdrop-blur-sm rounded-xl border-2 p-2 lg:p-4 w-4/5 max-w-xs shadow-lg transition-all duration-200
                ${
                  isDraftable
                    ? "border-blue-500 hover:border-blue-600 cursor-pointer transform hover:scale-105 active:scale-95 hover:shadow-xl"
                    : "border-gray-300 opacity-50"
                }`}
            >
              <div className="text-xs font-semibold leading-relaxed text-gray-800 break-words lg:text-sm">
                {formatCardText(tasks[taskIdx].desc)}
              </div>
              <div className="mt-2 text-xs sm:text-[10px] font-medium text-gray-600">
                Difficulty: {tasks[taskIdx].difficulty}
              </div>
            </div>
          );
        })}
        {isCurrentTurn && validActions && (
          <div className="flex justify-center">
            {validActions.some((action) => action.type === "nodraft") && (
              <button
                onClick={() => {
                  const passAction = validActions?.find(
                    (action) => action.type === "nodraft"
                  );
                  if (passAction) onDraft(passAction);
                }}
                className="px-4 py-2 text-sm font-semibold text-white bg-gray-600 border border-gray-400 rounded-lg sm:px-6 sm:py-3 button-hover focus:outline-none"
              >
                Pass
              </button>
            )}
          </div>
        )}
      </div>
    );
  }

  // if (phase === "end") {
  //   return (
  //     <div className="flex flex-col items-center justify-center h-full">
  //       <div
  //         className={`text-4xl font-bold mb-4 text-center ${
  //           win ? "text-green-600" : "text-red-600"
  //         }`}
  //       >
  //         {win ? "Victory" : "Defeat"}
  //       </div>
  //     </div>
  //   );
  // }

  return (
    <>
      {activeCards.map(([card, playerIdx], idx) => {
        const position = getPlayerPosition(playerIdx, currentPlayerIdx);
        const offset = {
          top:
            position === "top"
              ? "-mt-16 sm:-mt-20 md:-mt-24"
              : position === "bottom"
              ? "mt-16 sm:mt-20 md:mt-24"
              : "",
          left:
            position === "left"
              ? "-ml-16 sm:-ml-20 md:-ml-24"
              : position === "right"
              ? "ml-16 sm:ml-20 md:ml-24"
              : "",
        };

        return (
          <div
            key={idx}
            className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 
                w-[40px] h-[60px] md:w-16 md:h-24
                ${offset.top} ${offset.left}
                ${
                  SUIT_COLORS[card.suit as keyof typeof SUIT_COLORS]
                } rounded-lg border-2 flex items-center justify-center shadow-lg border-white`}
          >
            <div className="flex flex-col items-center">
              <span className="text-base font-bold text-white sm:text-lg">
                {card.rank}
              </span>
            </div>
          </div>
        );
      })}
    </>
  );
};
