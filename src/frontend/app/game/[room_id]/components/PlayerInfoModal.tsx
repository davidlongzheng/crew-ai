import { Card, card_to_string, Signal } from "@/lib/types";
import { formatCardText } from "../utils";

export interface PlayerInfoModalProps {
  tasks: Array<{ desc: string; status: string }>;
  signal?: Signal | null;
  hand: Card[];
}

export const PlayerInfoModal = ({
  tasks,
  signal,
  hand,
}: PlayerInfoModalProps) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
      <div className="min-w-[40%] w-[60%] p-4 border-2 rounded-lg shadow-xl pointer-events-auto bg-white/95 backdrop-blur-sm border-black min-h-[30%] overflow-y-auto">
        <ul className="space-y-2">
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
              className={`text-xs lg:text-sm font-medium ${
                item.status === "success"
                  ? "text-green-600"
                  : item.status === "fail"
                  ? "text-red-600"
                  : item.status === "unplayed"
                  ? "text-yellow-600"
                  : item.status === "played"
                  ? "text-gray-400"
                  : "text-gray-800"
              } border-b border-gray-200 last:border-b-0 pb-2`}
            >
              {formatCardText(item.desc)}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
