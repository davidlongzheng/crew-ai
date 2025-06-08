import { Card } from "@/lib/types";

// Modern suit colors that match the light blue aesthetic
const SUIT_COLORS = {
  0: "bg-blue-500", // blue
  1: "bg-green-500", // green
  2: "bg-pink-500", // pink
  3: "bg-yellow-500", // yellow
  4: "bg-gray-800", // trump
} as const;

export interface CardProps {
  card: Card;
  isPlayable?: boolean;
  onClick?: () => void;
  isAnimating?: boolean;
}

export const CardBox = ({
  card,
  isPlayable,
  onClick,
  isAnimating,
}: CardProps) => {
  const suitColor = SUIT_COLORS[card.suit as keyof typeof SUIT_COLORS];

  return (
    <div
      onClick={onClick}
      className={`
        w-12 h-16 sm:w-16 sm:h-24 
        ${suitColor} 
        rounded-lg shadow-lg backdrop-blur-sm 
        flex items-center justify-center
        transition-all duration-300 ease-out
        min-w-[48px] min-h-[64px]
        ${
          isPlayable
            ? "cursor-pointer hover:shadow-xl hover:-translate-y-2 hover:scale-105 active:scale-95 focus:outline-none focus:ring-4 focus:ring-blue-200"
            : "opacity-60 cursor-not-allowed"
        }
        ${
          card.is_trump
            ? "ring-2 ring-yellow-400 ring-opacity-75"
            : "border border-white/30"
        }
        ${isAnimating ? "animate-fadeInUp" : ""}`}
    >
      <div className="flex flex-col items-center">
        <span className="text-sm font-semibold text-white sm:text-lg drop-shadow-sm">
          {card.rank}
        </span>
      </div>
    </div>
  );
};
