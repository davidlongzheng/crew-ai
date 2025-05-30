"use client";

import { Dispatch, SetStateAction, useEffect, useRef, useState } from "react";
import useWebSocket, { ReadyState } from "react-use-websocket";
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
const LoadingScreen = () => (
  <main className="min-h-screen flex flex-col items-center justify-center bg-[#2c3e50] relative overflow-hidden">
    <div className="absolute inset-0 bg-[url('/pixel-pattern.svg')] opacity-10"></div>
    <div className="text-center space-y-8 relative z-10">
      <h1 className="text-4xl font-['Press_Start_2P'] text-white mb-4 pixel-text-shadow animate-pulse">
        Loading Game...
      </h1>
      <div className="w-16 h-16 border-4 border-white border-t-transparent animate-spin"></div>
    </div>
  </main>
);

const ErrorScreen = ({
  message,
  onReturnHome,
}: {
  message: string;
  onReturnHome: () => void;
}) => (
  <main className="min-h-screen flex flex-col items-center justify-center bg-[#2c3e50] relative overflow-hidden">
    <div className="absolute inset-0 bg-[url('/pixel-pattern.svg')] opacity-10"></div>
    <div className="text-center space-y-8 relative z-10">
      <h1 className="text-4xl font-['Press_Start_2P'] text-white mb-4 pixel-text-shadow">
        Error
      </h1>
      <p className="text-xl text-white/90 max-w-2xl font-['Press_Start_2P'] text-sm leading-relaxed">
        {message}
      </p>
      <button
        onClick={onReturnHome}
        className="px-8 py-4 bg-[#e74c3c] text-white rounded-none text-xl font-['Press_Start_2P'] 
                 hover:bg-[#c0392b] transition-colors duration-200 
                 border-4 border-white pixel-button-shadow
                 transform hover:scale-105 active:scale-95"
      >
        Return to Home
      </button>
    </div>
  </main>
);

// Custom hook for message processing
function processMessage(
  message: ServerMessage,
  gameState: GameState | null,
  setGameState: Dispatch<SetStateAction<GameState | null>>,
  sendJsonMessage: (message: ClientMessage) => void,
  roomId: string,
  messageQueue: Queue<ServerMessage>
) {
  // Handle sequence numbers
  if ("seqnum" in message) {
    if (gameState) {
      if (gameState.seqnum >= message.seqnum!) {
        return;
      }
      if (message.seqnum !== gameState.seqnum + 1) {
        alert("Oops something went wrong. Please refresh the page.");
        return;
      }
    } else if (message.type !== "full_state") {
      messageQueue.enqueue(message);
      return;
    }
  }

  switch (message.type) {
    case "connect_ack":
      sendJsonMessage({ type: "full_sync", room_id: roomId });
      break;

    case "full_state":
      setGameState({
        seqnum: message.seqnum!,
        stage: message.stage!,
        player_uids: message.player_uids!,
        handles: message.handles!,
        players: message.players!,
        cur_uid: message.cur_uid ?? null,
        engine_state: message.engine_state ?? null,
        valid_actions: message.valid_actions ?? null,
        tasks: message.tasks ?? null,
        num_players: message.num_players!,
        difficulty: message.difficulty!,
      });
      break;

    case "settings_updated":
      setGameState((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          seqnum: message.seqnum!,
          num_players: message.num_players ?? prev.num_players,
          difficulty: message.difficulty ?? prev.difficulty,
        };
      });
      break;

    case "joined_game":
      setGameState((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          seqnum: message.seqnum!,
          player_uids: [...prev.player_uids, message.uid!],
          handles: [...prev.handles, message.handle!],
        };
      });
      break;

    case "started_game":
      setGameState((prev) => {
        if (!prev) return null;
        return {
          ...prev,
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
        if (!prev) return null;
        return {
          ...prev,
          seqnum: message.seqnum!,
          cur_uid: message.cur_uid ?? null,
          engine_state: message.engine_state!,
          valid_actions: message.valid_actions ?? null,
        };
      });
      break;

    case "trick_won":
      setGameState((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          seqnum: message.seqnum!,
        };
      });
      break;

    case "kicked_player":
      setGameState((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          seqnum: message.seqnum!,
          handles: prev.handles.filter((h) => h !== message.handle),
          player_uids: prev.player_uids.filter((uid) => uid !== message.uid),
        };
      });
      break;

    case "ended_game":
      setGameState((prev) => {
        if (!prev) return null;
        return {
          ...prev,
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
  const { sendJsonMessage, lastJsonMessage, readyState } = useWebSocket(
    `ws://localhost:8000/game/ws/${roomId}?uid=${uid}`
  );
  const messageQueue = useRef<Queue<ServerMessage>>(new Queue());
  const gameStateRef = useRef(gameState);

  useEffect(() => {
    gameStateRef.current = gameState;
  }, [gameState]);

  useEffect(() => {
    if (!lastJsonMessage || !roomId) return;

    const message = lastJsonMessage as ServerMessage;
    processMessage(
      message,
      gameStateRef.current,
      setGameState,
      sendJsonMessage,
      roomId,
      messageQueue.current
    );

    // Process any queued messages after gameState is set
    while (gameStateRef.current && messageQueue.current.size() > 0) {
      processMessage(
        messageQueue.current.dequeue()!,
        gameStateRef.current,
        setGameState,
        sendJsonMessage,
        roomId,
        messageQueue.current
      );
    }
  }, [roomId, lastJsonMessage, sendJsonMessage]);

  if (readyState === ReadyState.CLOSED) {
    return (
      <ErrorScreen
        message="Unable to connect to the game server. Please go back to the front page and try again."
        onReturnHome={() => (window.location.href = "/")}
      />
    );
  }

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

        const response = await fetch("http://localhost:8000/game/set_uid", {
          method: "POST",
        });
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
      <ErrorScreen
        message="Missing required game parameters. Please return to the home page and try again."
        onReturnHome={() => (window.location.href = "/")}
      />
    );
  }

  return <GamePage uid={uid} roomId={roomId} />;
}
