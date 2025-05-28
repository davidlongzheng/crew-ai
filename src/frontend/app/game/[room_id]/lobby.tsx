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
    <main className="min-h-screen p-8 bg-[#2c3e50] relative overflow-hidden">
      {/* Pixelated background pattern */}
      <div className="absolute inset-0 bg-[url('/pixel-pattern.svg')] opacity-10"></div>

      <div className="relative z-10 max-w-[627px] mx-auto">
        <h1 className="text-4xl font-['Press_Start_2P'] text-white mb-8 pixel-text-shadow">
          Game Lobby
        </h1>

        <div className="mb-8 flex gap-8 items-center bg-[#34495e] p-6 rounded-none border-4 border-white">
          <div>
            <label
              htmlFor="numPlayers"
              className="block text-sm font-['Press_Start_2P'] text-white mb-2"
            >
              Number of Players:
            </label>
            <select
              id="numPlayers"
              value={gameState.num_players}
              onChange={(e) => handleSetSettings(Number(e.target.value))}
              className="border-4 border-white bg-[#2c3e50] text-white p-2 font-['Press_Start_2P'] text-sm"
            >
              {[2, 3, 4, 5].map((n) => (
                <option
                  key={n}
                  value={n}
                  disabled={gameState.handles.length > n}
                >
                  {n}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label
              htmlFor="difficulty"
              className="block text-sm font-['Press_Start_2P'] text-white mb-2"
            >
              Difficulty:
            </label>
            <select
              id="difficulty"
              value={gameState.difficulty}
              onChange={(e) =>
                handleSetSettings(undefined, Number(e.target.value))
              }
              className="border-4 border-white bg-[#2c3e50] text-white p-2 font-['Press_Start_2P'] text-sm"
            >
              {[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              ].map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="mb-8 bg-[#34495e] p-6 rounded-none border-4 border-white">
          <h2 className="text-2xl font-['Press_Start_2P'] text-white mb-4">
            Players:
          </h2>
          <ul className="space-y-2">
            {gameState.handles.map((h, i) => (
              <li
                key={i}
                className="flex items-center justify-between text-white font-['Press_Start_2P'] text-sm"
              >
                <span>
                  {h} {gameState.player_uids[i] == uid && "(me)"}
                </span>
                <button
                  onClick={() => {
                    sendJsonMessage({
                      type: "kick_player",
                      room_id: roomId,
                      handle: h,
                    });
                  }}
                  className="ml-2 text-red-500 hover:text-red-700 font-['Press_Start_2P']"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </div>

        {!joined && gameState.handles.length < gameState.num_players && (
          <div className="mb-8 bg-[#34495e] p-6 rounded-none border-4 border-white">
            <input
              type="text"
              value={handle}
              onChange={(e) => setHandle(e.target.value)}
              placeholder="Enter your handle"
              className="border-4 border-white bg-[#2c3e50] text-white p-2 mr-4 font-['Press_Start_2P'] text-sm"
            />
            <button
              onClick={handleJoinGame}
              className="bg-[#e74c3c] text-white px-6 py-2 rounded-none font-['Press_Start_2P'] text-sm
                       hover:bg-[#c0392b] transition-colors duration-200 border-4 border-white
                       transform hover:scale-105 active:scale-95"
            >
              Join Game
            </button>
          </div>
        )}

        {gameState.handles.length < gameState.num_players && (
          <div className="mb-8">
            <button
              onClick={() => {
                sendJsonMessage({
                  type: "add_ai",
                  room_id: roomId,
                });
              }}
              className="bg-[#9b59b6] text-white px-6 py-2 rounded-none font-['Press_Start_2P'] text-sm
                       hover:bg-[#8e44ad] transition-colors duration-200 border-4 border-white
                       transform hover:scale-105 active:scale-95"
            >
              Add AI
            </button>
          </div>
        )}

        {gameState.handles.length === gameState.num_players ? (
          <button
            onClick={handleStartGame}
            className="bg-[#27ae60] text-white px-8 py-4 rounded-none font-['Press_Start_2P'] text-lg
                     hover:bg-[#219a52] transition-colors duration-200 border-4 border-white
                     transform hover:scale-105 active:scale-95"
          >
            Start Game
          </button>
        ) : (
          <div className="text-white font-['Press_Start_2P'] text-lg">
            Waiting for {gameState.num_players - gameState.handles.length}{" "}
            players to join...
          </div>
        )}
      </div>
    </main>
  );
}
