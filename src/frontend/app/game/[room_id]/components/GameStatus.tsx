export interface GameStatusProps {
  phase: string;
  trick: number;
}

export const GameStatus = ({ phase, trick }: GameStatusProps) => (
  <div className="absolute z-10 px-6 py-3 -translate-x-1/2 border shadow-lg top-4 left-1/2 bg-white/90 backdrop-blur-sm rounded-xl border-white/30">
    <span className="text-sm font-semibold text-gray-800">
      {phase === "draft"
        ? `Draft Round ${trick + 1}`
        : phase === "end"
        ? "Game Over"
        : `Trick ${trick + 1} - ${phase == "play" ? "Play" : "Signal"}`}
    </span>
  </div>
);
