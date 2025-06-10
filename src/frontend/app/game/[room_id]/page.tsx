"use client";

import { Dispatch, SetStateAction, useEffect, useRef, useState } from "react";
import useWebSocket from "react-use-websocket";
import { ClientMessage, GameState, ServerMessage } from "@/lib/types";
import { Queue } from "@/lib/queue";
import { useParams } from "next/navigation";
import { LobbyStage } from "./lobby";
import { PlayStage } from "./play";

// Types
interface GamePageProps {
  uid: string;
  roomId: string;
}

// Components
const LoadingScreen = () => {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen px-4 sm:px-6 lg:px-8">
      <div className="max-w-md mx-auto space-y-8 text-center animate-fadeInUp">
        <div className="space-y-4">
          <div className="w-16 h-16 mx-auto border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
          <h1 className="text-2xl font-semibold text-gray-800 sm:text-3xl">
            Loading Game...
          </h1>
          <button
            onClick={() => {
              window.location.href = "/";
            }}
            className="w-full sm:w-auto px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold 
                   button-hover focus:outline-none focus:ring-4 focus:ring-blue-200
                   min-w-[200px]"
          >
            Return to Home
          </button>
        </div>
      </div>
    </main>
  );
};

const ErrorScreen = ({ message }: { message: string }) => (
  <main className="flex flex-col items-center justify-center min-h-screen px-4 sm:px-6 lg:px-8">
    <div className="max-w-md mx-auto space-y-8 text-center animate-fadeInUp">
      <div className="space-y-6">
        {/* Error Icon */}
        <div className="flex items-center justify-center w-16 h-16 mx-auto bg-red-100 rounded-full">
          <svg
            className="w-8 h-8 text-red-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
            />
          </svg>
        </div>

        <div className="space-y-2">
          <h1 className="text-2xl font-semibold text-gray-800 sm:text-3xl">
            Connection Error
          </h1>
          <p className="font-medium leading-relaxed text-gray-600">{message}</p>
        </div>
      </div>

      <div className="space-y-4">
        <button
          onClick={() => {
            window.location.href = "/";
          }}
          className="w-full sm:w-auto px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold 
                   button-hover focus:outline-none focus:ring-4 focus:ring-blue-200
                   min-w-[200px]"
        >
          Return to Home
        </button>
        <p className="text-sm text-gray-500">
          Try refreshing the page or creating a new game room
        </p>
      </div>
    </div>
  </main>
);

function processUnsequencedMessage(
  message: ServerMessage,
  setGameState: Dispatch<SetStateAction<GameState | null>>,
  sendJsonMessage: (message: ClientMessage) => void,
  roomId: string
) {
  switch (message.type) {
    case "connect_ack":
      sendJsonMessage({ type: "full_sync", room_id: roomId });
      break;

    case "full_state":
      setGameState({
        seqnum: message.start_seqnum!,
        stage: message.stage!,
        player_uids: message.player_uids!,
        handles: message.handles!,
        players: message.players!,
        cur_uid: message.cur_uid ?? null,
        engine_state: message.engine_state ?? null,
        valid_actions: message.valid_actions ?? null,
        active_cards: message.active_cards!,
        tasks: message.tasks ?? null,
        num_players: message.num_players!,
        difficulty: message.difficulty!,
      });
      break;
    case "error":
      alert(message.message!);
      break;
  }
}

// Custom hook for message processing
function processSequencedMessage(
  message: ServerMessage,
  gameState: GameState,
  setGameState: Dispatch<SetStateAction<GameState | null>>
) {
  // Handle sequence numbers
  if (gameState.seqnum >= message.seqnum!) {
    return;
  }
  if (message.seqnum !== gameState.seqnum + 1) {
    alert("Oops something went wrong. Please refresh the page.");
    return;
  }

  switch (message.type) {
    case "settings_updated":
      setGameState((prev) => {
        return {
          ...prev!,
          seqnum: message.seqnum!,
          num_players: message.num_players ?? prev!.num_players,
          difficulty: message.difficulty ?? prev!.difficulty,
        };
      });
      break;

    case "joined_game":
      setGameState((prev) => {
        return {
          ...prev!,
          seqnum: message.seqnum!,
          player_uids: [...prev!.player_uids, message.uid!],
          handles: [...prev!.handles, message.handle!],
        };
      });
      break;

    case "started_game":
      setGameState((prev) => {
        return {
          ...prev!,
          seqnum: message.seqnum!,
          stage: "play",
          players: message.players!,
          cur_uid: message.cur_uid!,
          engine_state: message.engine_state!,
          valid_actions: message.valid_actions!,
          tasks: message.tasks!,
        };
      });
      break;

    case "moved":
      setGameState((prev) => {
        return {
          ...prev!,
          seqnum: message.seqnum!,
          cur_uid: message.cur_uid ?? null,
          engine_state: message.engine_state!,
          valid_actions: message.valid_actions ?? null,
          active_cards: message.active_cards!,
        };
      });
      break;

    case "kicked_player":
      setGameState((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          seqnum: message.seqnum!,
          handles: prev!.handles.filter((h) => h !== message.handle),
          player_uids: prev!.player_uids.filter((uid) => uid !== message.uid),
        };
      });
      break;

    case "ended_game":
      setGameState((prev) => {
        return {
          ...prev!,
          seqnum: message.seqnum!,
          stage: "lobby",
          players: [],
        };
      });
      break;
  }
}

// Main component
const GamePage = ({ uid, roomId }: GamePageProps) => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const { sendJsonMessage, lastJsonMessage } = useWebSocket(
    `${process.env.NEXT_PUBLIC_API_WS_URL}/game/ws/${roomId}?uid=${uid}`,
    {
      shouldReconnect: (closeEvent) => {
        // Optional: you can inspect closeEvent here
        console.warn(
          `WebSocket closed with code ${closeEvent.code}, reason: ${closeEvent.reason}.`
        );
        return true; // Always try to reconnect
      },
      onClose: () => {
        console.warn("WebSocket closed, attempting to reconnect...");
      },
      onError: (event) => {
        console.error("WebSocket error:", event);
      },
    }
  );
  const messageQueue = useRef<Queue<ServerMessage>>(new Queue());
  const gameStateRef = useRef(gameState);

  useEffect(() => {
    gameStateRef.current = gameState;
  }, [gameState]);

  useEffect(() => {
    if (!lastJsonMessage || !roomId) return;

    const message = lastJsonMessage as ServerMessage;
    if (!("seqnum" in message)) {
      processUnsequencedMessage(message, setGameState, sendJsonMessage, roomId);
    } else {
      messageQueue.current.enqueue(message);
    }

    // Process any queued messages after gameState is set
    while (gameStateRef.current && messageQueue.current.size() > 0) {
      processSequencedMessage(
        messageQueue.current.dequeue()!,
        gameStateRef.current!,
        setGameState
      );
    }
  }, [roomId, lastJsonMessage, sendJsonMessage]);

  if (!gameState) {
    return <LoadingScreen />;
  }

  if (gameState.stage === "lobby") {
    return (
      <LobbyStage
        gameState={gameState}
        uid={uid}
        roomId={roomId}
        sendJsonMessage={sendJsonMessage}
      />
    );
  }

  return (
    <PlayStage
      gameState={gameState}
      uid={uid}
      roomId={roomId}
      sendJsonMessage={sendJsonMessage}
    />
  );
};

// Wrapper component for loading UID
export default function GamePageWithLoading() {
  const params = useParams();
  const roomId = params.room_id as string;
  const [uid, setUid] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const getUid = async () => {
      try {
        const cookies = document.cookie.split(";");
        const uidCookie = cookies.find((cookie) =>
          cookie.trim().startsWith("uid=")
        );

        if (uidCookie) {
          setUid(uidCookie.split("=")[1]);
          return;
        }

        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/game/set_uid`,
          {
            method: "POST",
          }
        );
        const data = await response.json();
        document.cookie = `uid=${data.uid}`;
        setUid(data.uid);
      } catch (error) {
        console.error("Failed to set uid:", error);
      } finally {
        setIsLoading(false);
      }
    };

    getUid();
  }, []);

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!uid || !roomId) {
    return (
      <ErrorScreen message="Missing required game parameters. Please return to the home page and try again." />
    );
  }

  return <GamePage uid={uid} roomId={roomId} />;
}
