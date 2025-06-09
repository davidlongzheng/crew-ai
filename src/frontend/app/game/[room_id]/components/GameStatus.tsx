export interface GameStatusProps {
  phase: string;
  trick: number;
}

export const GameStatus = ({ phase, trick }: GameStatusProps) => {
  if (phase === "end") {
    return <div></div>;
  }

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
      <span className="text-sm font-semibold text-white">
        {phase === "draft"
          ? `Draft Round ${trick + 1}`
          : `Trick ${trick + 1} - ${phase == "play" ? "Play" : "Signal"}`}
      </span>
    </div>
  );
};
