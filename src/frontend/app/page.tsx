"use client";

import { useState } from "react";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);

  const handleCreateRoom = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8000/game/create_room", {
        method: "POST",
      });
      const data = await response.json();
      window.location.href = `/game/${data.room_id}`;
    } catch (error) {
      console.error("Failed to create room:", error);
      setIsLoading(false);
    }
  };

  // Predefined star positions and delays
  const stars = [
    { left: "10%", top: "20%", delay: "0.2s" },
    { left: "20%", top: "40%", delay: "0.4s" },
    { left: "30%", top: "60%", delay: "0.6s" },
    { left: "40%", top: "80%", delay: "0.8s" },
    { left: "50%", top: "30%", delay: "1.0s" },
    { left: "60%", top: "50%", delay: "1.2s" },
    { left: "70%", top: "70%", delay: "1.4s" },
    { left: "80%", top: "90%", delay: "1.6s" },
    { left: "90%", top: "10%", delay: "1.8s" },
    { left: "15%", top: "30%", delay: "0.3s" },
    { left: "25%", top: "50%", delay: "0.5s" },
    { left: "35%", top: "70%", delay: "0.7s" },
    { left: "45%", top: "90%", delay: "0.9s" },
    { left: "55%", top: "20%", delay: "1.1s" },
    { left: "65%", top: "40%", delay: "1.3s" },
    { left: "75%", top: "60%", delay: "1.5s" },
    { left: "85%", top: "80%", delay: "1.7s" },
    { left: "95%", top: "10%", delay: "1.9s" },
    { left: "5%", top: "40%", delay: "0.1s" },
    { left: "15%", top: "60%", delay: "0.3s" },
  ];

  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-[#2c3e50] relative overflow-hidden">
      {/* Pixelated background pattern */}
      <div className="absolute inset-0 bg-[url('/pixel-pattern.svg')] opacity-10"></div>

      {/* 8-bit stars */}
      <div className="absolute inset-0">
        {stars.map((star, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 bg-yellow-300 animate-twinkle"
            style={{
              left: star.left,
              top: star.top,
              animationDelay: star.delay,
            }}
          />
        ))}
      </div>

      <div className="text-center space-y-8 relative z-10">
        <h1 className="text-6xl font-['Press_Start_2P'] text-white mb-4 pixel-text-shadow">
          Welcome to Crew Game
        </h1>
        <p className="text-xl text-white/90 max-w-2xl font-['Press_Start_2P'] text-sm leading-relaxed">
          Create a new game room and invite your friends to play together!
        </p>
        <button
          onClick={handleCreateRoom}
          disabled={isLoading}
          className="px-8 py-4 bg-[#e74c3c] text-white rounded-none text-xl font-['Press_Start_2P'] 
                   hover:bg-[#c0392b] transition-colors duration-200 
                   border-4 border-white pixel-button-shadow
                   disabled:opacity-50 disabled:cursor-not-allowed
                   transform hover:scale-105 active:scale-95"
        >
          {isLoading ? "Creating Room..." : "Create New Game Room"}
        </button>
      </div>
    </main>
  );
}
