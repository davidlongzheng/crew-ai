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
}

export const CardBox = ({ card, isPlayable, onClick }: CardProps) => {
  const suitColor = SUIT_COLORS[card.suit as keyof typeof SUIT_COLORS];

  return (
    <div
      onClick={onClick}
      className={`
        w-[40px] h-[60px] sm:w-16 sm:h-24 
        ${suitColor} 
        rounded-lg border-2 shadow-lg backdrop-blur-sm 
        flex items-center justify-center border-white
        ${
          isPlayable
            ? "cursor-pointer sm:hover:shadow-xl sm:hover:-translate-y-2 sm:hover:scale-105"
            : "opacity-60 cursor-not-allowed"
        }`}
    >
      <div className="flex flex-col items-center">
        <span className="text-sm font-semibold text-white sm:text-lg drop-shadow-sm">
          {card.rank}
        </span>
      </div>
    </div>
  );
};
