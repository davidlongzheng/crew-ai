"use client";

import { useState } from "react";
import { ClientMessage, GameState } from "@/lib/types";

interface LobbyStageProps {
  gameState: GameState;
  uid: string;
  roomId: string;
  sendJsonMessage: (message: ClientMessage) => void;
}

export function LobbyStage({
  gameState,
  uid,
  roomId,
  sendJsonMessage,
}: LobbyStageProps) {
  const [handle, setHandle] = useState("");

  const handleJoinGame = () => {
    if (!handle) return;
    if (gameState.handles.length >= gameState.num_players) {
      alert(`Cannot join - game is full (${gameState.num_players} players)`);
      return;
    }
    sendJsonMessage({
      type: "join_game",
      room_id: roomId,
      handle,
    });
  };

  const handleStartGame = () => {
    sendJsonMessage({
      type: "start_game",
      room_id: roomId,
      num_players: gameState.num_players,
      difficulty: gameState.difficulty,
    });
  };

  const handleSetSettings = (numPlayers?: number, difficulty?: number) => {
    sendJsonMessage({
      type: "set_settings",
      room_id: roomId,
      num_players: numPlayers,
      difficulty: difficulty,
    });
  };

  const joined = gameState.player_uids.includes(uid);

  return (
    <main className="min-h-screen px-4 py-8 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto space-y-8 animate-fadeInUp">
        {/* Header */}
        <div className="space-y-4 text-center">
          <h1 className="text-3xl font-bold text-gray-800 sm:text-4xl">
            Game Lobby
          </h1>
          <p className="font-medium text-gray-600">
            Configure game settings and share room link with friends!
          </p>
        </div>

        {/* Game Settings */}
        <div className="p-6 shadow-lg bg-white/80 backdrop-blur-sm rounded-xl">
          <h2 className="mb-6 text-xl font-semibold text-gray-800">
            Game Settings
          </h2>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div className="space-y-2">
              <label
                htmlFor="numPlayers"
                className="block text-sm font-semibold text-gray-700"
              >
                Number of Players:
              </label>
              <select
                id="numPlayers"
                value={gameState.num_players}
                onChange={(e) => handleSetSettings(Number(e.target.value))}
                className="w-full px-4 py-2 font-medium border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {[3, 4].map((n) => (
                  <option
                    key={n}
                    value={n}
                    disabled={gameState.handles.length > n}
                  >
                    {n} Players
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label
                htmlFor="difficulty"
                className="block text-sm font-semibold text-gray-700"
              >
                Difficulty Level:
              </label>
              <select
                id="difficulty"
                value={gameState.difficulty}
                onChange={(e) =>
                  handleSetSettings(undefined, Number(e.target.value))
                }
                className="w-full px-4 py-2 font-medium border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {[
                  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                ].map((d) => (
                  <option key={d} value={d}>
                    Level {d}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Players List */}
        <div className="p-6 shadow-lg bg-white/80 backdrop-blur-sm rounded-xl">
          <h2 className="mb-6 text-xl font-semibold text-gray-800">
            Players ({gameState.handles.length}/{gameState.num_players})
          </h2>
          <div className="space-y-3">
            {gameState.handles.map((h, i) => {
              const isAI = gameState.player_uids[i]?.startsWith("ai_");
              const isCurrentUser = gameState.player_uids[i] == uid;

              return (
                <div
                  key={i}
                  className="flex items-center justify-between p-4 rounded-lg shadow-md bg-gray-50"
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        isAI ? "bg-purple-100" : "bg-blue-100"
                      }`}
                    >
                      {isAI ? (
                        <svg
                          className="w-4 h-4 text-purple-600"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                          />
                        </svg>
                      ) : (
                        <svg
                          className="w-4 h-4 text-blue-600"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-gray-800">{h}</span>
                      {isCurrentUser && (
                        <span className="font-medium text-blue-600">(you)</span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      sendJsonMessage({
                        type: "kick_player",
                        room_id: roomId,
                        handle: h,
                      });
                    }}
                    className="p-2 text-red-500 transition-colors rounded-lg hover:text-red-700 hover:bg-red-50"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Join Game */}
        {!joined && gameState.handles.length < gameState.num_players && (
          <div className="p-6 shadow-lg bg-white/80 backdrop-blur-sm rounded-xl">
            <h2 className="mb-4 text-xl font-semibold text-gray-800">
              Join the Game
            </h2>
            <div className="flex flex-col gap-4 sm:flex-row">
              <input
                type="text"
                value={handle}
                onChange={(e) => setHandle(e.target.value)}
                placeholder="Enter your username"
                className="flex-1 px-4 py-3 font-medium border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <button
                onClick={handleJoinGame}
                className="px-6 py-3 font-semibold text-white bg-blue-600 rounded-lg button-hover focus:outline-none focus:ring-4 focus:ring-blue-200"
              >
                Join Game
              </button>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-col justify-center gap-4 sm:flex-row">
          {gameState.handles.length < gameState.num_players && (
            <button
              onClick={() => {
                sendJsonMessage({
                  type: "add_ai",
                  room_id: roomId,
                });
              }}
              className="px-6 py-3 font-semibold text-white bg-purple-600 rounded-lg button-hover focus:outline-none focus:ring-4 focus:ring-purple-200"
            >
              Add AI Player
            </button>
          )}

          {gameState.handles.length === gameState.num_players ? (
            <button
              onClick={handleStartGame}
              className="px-8 py-4 text-lg font-semibold text-white bg-green-600 rounded-lg button-hover focus:outline-none focus:ring-4 focus:ring-green-200"
            >
              Start Game
            </button>
          ) : (
            <div className="p-4 text-center">
              <div className="inline-flex items-center space-x-2 text-gray-600">
                <div className="w-4 h-4 border-2 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
                <span className="font-medium">
                  Waiting for {gameState.num_players - gameState.handles.length}{" "}
                  more player
                  {gameState.num_players - gameState.handles.length !== 1
                    ? "s"
                    : ""}
                  ...
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
