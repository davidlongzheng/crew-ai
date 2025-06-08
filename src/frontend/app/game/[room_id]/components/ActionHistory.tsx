import { Action, action_to_string } from "@/lib/types";
import { formatCardText } from "../utils";

export interface ActionHistoryProps {
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

export const ActionHistory = ({
  history,
  handles,
  players,
  tasks,
}: ActionHistoryProps) => (
  <div
    className="flex-1 space-y-3 overflow-y-auto"
    ref={(el) => el?.scrollTo(0, el.scrollHeight)}
  >
    {history.map((event, idx) => (
      <div
        key={idx}
        className="p-3 text-sm border border-gray-200 rounded-lg shadow-sm bg-gray-50"
      >
        {event.type === "action" && event.action && (
          <>
            <span className="font-semibold text-blue-600">
              {handles[players.indexOf(event.action.player)]}:{" "}
            </span>
            <span className="font-medium text-gray-800">
              {formatCardText(action_to_string(event.action, tasks))}
            </span>
          </>
        )}
        {event.type === "trick_winner" && event.trick_winner !== null && (
          <span className="font-semibold text-green-600">
            Trick {event.trick! + 1} won by{" "}
            {handles[players.indexOf(event.trick_winner!)]}
          </span>
        )}
        {event.type === "new_trick" && (
          <span className="font-semibold text-purple-600">
            Starting {event.phase === "draft" ? "Draft Round " : "Trick "}
            {event.trick! + 1}
          </span>
        )}
        {event.type === "game_ended" && (
          <span className="font-semibold text-red-600">Game Over</span>
        )}
      </div>
    ))}
  </div>
);
