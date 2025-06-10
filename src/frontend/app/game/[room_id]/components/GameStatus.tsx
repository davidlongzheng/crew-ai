export interface GameStatusProps {
  phase: string;
  trick: number;
  win: boolean;
}

export const GameStatus = ({ phase, trick, win }: GameStatusProps) => {
  return (
    <div
      className={`absolute top-4 left-1/2 -translate-x-1/2 px-3 py-2 min-w-[140px] border shadow-lg backdrop-blur-sm rounded-xl border-gray-400 text-center ${
        phase === "play"
          ? "bg-blue-500/90"
          : phase === "draft"
          ? "bg-green-500/90"
          : phase == "signal"
          ? "bg-yellow-500/90"
          : "bg-white"
      }`}
    >
      <span
        className={`text-med font-semibold ${
          phase === "end"
            ? win
              ? "text-green-500"
              : "text-red-500"
            : "text-white"
        }`}
      >
        {phase === "draft"
          ? `Draft Round ${trick + 1}`
          : phase === "end"
          ? win
            ? "Victory"
            : "Defeat"
          : `Trick ${trick + 1} - ${phase == "play" ? "Play" : "Signal"}`}
      </span>
    </div>
  );
};
