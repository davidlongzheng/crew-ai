import { Action } from "@/lib/types";
import { CardBox } from "./CardBox";

export interface PlayerHandProps {
  phase: string;
  isCurrentTurn: boolean;
  validActions: Action[] | null;
  hand: Array<{ rank: number; suit: number; is_trump: boolean }>;
  onMove: (action: Action) => void;
}

export const PlayerHand = ({
  phase,
  isCurrentTurn,
  validActions,
  hand,
  onMove,
}: PlayerHandProps) => {
  return (
    <div className="w-full max-w-full p-4 border shadow-lg bg-white/80 backdrop-blur-sm rounded-xl border-white/30 sm:p-6">
      {isCurrentTurn && validActions && (
        <div className="mb-4 text-sm font-medium text-center text-gray-800 sm:text-left">
          {phase === "draft"
            ? "It's your turn to draft! Click on one of the tasks above or pass."
            : phase === "signal"
            ? "It's your turn to signal! Click on one of the cards below or pass."
            : "It's your turn to play! Click on one of the cards below."}
        </div>
      )}

      <div className="flex flex-col gap-4">
        {/* Cards container with flex wrap */}
        <div className="flex flex-wrap justify-center gap-2 sm:gap-3 sm:justify-start">
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

        {/* Pass button */}
        {isCurrentTurn && validActions && (
          <div className="flex justify-center">
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
                className="px-4 py-2 text-sm font-semibold text-white bg-gray-600 border border-gray-400 rounded-lg sm:px-6 sm:py-3 button-hover focus:outline-none"
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
