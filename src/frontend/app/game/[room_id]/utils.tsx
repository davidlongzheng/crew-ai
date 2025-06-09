import { TO_SUIT_NUM } from "@/lib/types";
import { SUIT_COLORS } from "./constants";

export const formatCardText = (text: string) => {
  const cardRegex = /(\d[bgypt])/g;
  const parts = text.split(cardRegex);

  return parts.map((part, idx) => {
    if (cardRegex.test(part)) {
      const rank = part[0];
      const suit = TO_SUIT_NUM[part[1]];
      return (
        <span
          key={idx}
          className={`inline-flex items-center justify-center mx-0.5 ${
            SUIT_COLORS[suit as keyof typeof SUIT_COLORS]
          } rounded-lg shadow-lg backdrop-blur-sm border border-white px-1.5 py-0.5 min-w-[20px] h-5`}
        >
          <span className="text-xs font-semibold text-white drop-shadow-sm">
            {rank}
          </span>
        </span>
      );
    }
    return part;
  });
};

export type PlayerPosition = "top" | "bottom" | "left" | "right";

export const getPlayerPosition = (
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
