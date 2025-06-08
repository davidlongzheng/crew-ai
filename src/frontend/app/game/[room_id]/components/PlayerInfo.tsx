import { useState } from "react";
import { Card, card_to_string, Signal } from "@/lib/types";
import { formatCardText } from "../utils";

export interface PlayerInfoProps {
  handle: string;
  phase: string;
  isCaptain: boolean;
  isLeader: boolean;
  isCurrentTurn: boolean;
  tasks: Array<{ desc: string; status: string }>;
  signal?: Signal | null;
  hand: Card[];
}

export const PlayerInfo = ({
  handle,
  isCaptain,
  isLeader,
  isCurrentTurn,
  tasks,
  signal,
  hand,
}: PlayerInfoProps) => {
  const [showModal, setShowModal] = useState(false);

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

  const playerInfoModal = (tasks.length > 0 || signal) && showModal && (
    <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
      <div className="w-64 p-4 border rounded-lg shadow-xl pointer-events-auto bg-white/95 backdrop-blur-sm border-white/30">
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
              className={`text-xs font-medium ${
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

  const playerLabel = (
    <div
      className={`px-4 py-2 rounded-lg shadow-lg border cursor-pointer ${
        isCurrentTurn
          ? "bg-blue-500 border-blue-600 text-white"
          : "bg-white/90 backdrop-blur-sm border-white/30 text-gray-800"
      } ${
        isLeader ? "ring-2 ring-cyan-400 ring-opacity-75" : ""
      } font-semibold text-sm transition-all duration-200 hover:scale-105`}
      onClick={() => setShowModal(!showModal)}
    >
      <div className="flex items-center gap-2">
        <div className="flex items-center">
          {isCaptain && (
            <>
              <span className="flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-yellow-500 rounded-full">
                C
              </span>
              <span className="mr-1" />
            </>
          )}
          {handle}
        </div>

        {/* Status indicators */}
        {(successfulTasks > 0 ||
          failedTasks > 0 ||
          unresolvedTasks > 0 ||
          signalPlayed > 0 ||
          signalUnplayed > 0) && (
          <div className="flex gap-1">
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

  return (
    <>
      {playerLabel}
      {playerInfoModal}
    </>
  );
};
