import { useState } from "react";
import { Card, card_to_string, Signal } from "@/lib/types";
import { formatCardText } from "../utils";

export interface PlayerHandleProps {
  handle: string;
  phase: string;
  isCaptain: boolean;
  isCurrentTurn: boolean;
  tasks: Array<{ desc: string; status: string }>;
  signal?: Signal | null;
  hand: Card[];
}

export const PlayerHandle = ({
  handle,
  phase,
  isCaptain,
  isCurrentTurn,
  tasks,
  signal,
  hand,
}: PlayerHandleProps) => {
  // Calculate counts for indicators
  const successfulTasks = tasks.filter(
    (task) => task.status === "success"
  ).length;
  const failedTasks = tasks.filter((task) => task.status === "fail").length;
  const unresolvedTasks = tasks.filter(
    (task) => task.status !== "success" && task.status !== "fail"
  ).length;

  const signalPlayed =
    signal &&
    !hand?.some(
      (card) => card.rank === signal.card.rank && card.suit === signal.card.suit
    )
      ? 1
      : 0;
  const signalUnplayed =
    signal &&
    hand?.some(
      (card) => card.rank === signal.card.rank && card.suit === signal.card.suit
    )
      ? 1
      : 0;

  return (
    <div
      className={`px-4 py-2 rounded-lg shadow-lg border border-gray-400 cursor-pointer ${
        isCurrentTurn
          ? `${
              phase === "play"
                ? "bg-blue-500"
                : phase === "draft"
                ? "bg-green-500"
                : "bg-yellow-500"
            } text-white`
          : "bg-white/90 backdrop-blur-s text-gray-800"
      } font-semibold text-sm sm:hover:scale-105`}
    >
      <div className="flex items-center gap-2">
        <div className="flex items-center">{handle}</div>

        {/* Status indicators */}
        {(successfulTasks > 0 ||
          failedTasks > 0 ||
          unresolvedTasks > 0 ||
          signalPlayed > 0 ||
          signalUnplayed > 0 ||
          isCaptain) && (
          <div className="flex gap-1">
            {isCaptain && (
              <span className="flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-yellow-600 rounded-full">
                C
              </span>
            )}
            {successfulTasks > 0 && (
              <span className="flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-green-500 rounded-full">
                {successfulTasks}
              </span>
            )}
            {failedTasks > 0 && (
              <span className="flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-red-500 rounded-full">
                {failedTasks}
              </span>
            )}
            {unresolvedTasks > 0 && (
              <span className="flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-gray-500 rounded-full">
                {unresolvedTasks}
              </span>
            )}
            {signalUnplayed > 0 && (
              <span className="flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-yellow-500 rounded-full">
                S
              </span>
            )}
            {signalPlayed > 0 && (
              <span className="flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-gray-400 rounded-full">
                S
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
