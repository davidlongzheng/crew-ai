"use client";

import { useState } from "react";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);

  const handleCreateRoom = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/game/create_room`,
        {
          method: "POST",
        }
      );
      const data = await response.json();
      window.location.href = `/game/${data.room_id}`;
    } catch (error) {
      console.error("Failed to create room:", error);
      setIsLoading(false);
    }
  };

  return (
    <main className="flex flex-col items-center justify-center px-4 min-h-[500px] sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto space-y-8 text-center animate-fadeInUp">
        {/* Header Section */}
        <div className="space-y-6">
          <h1 className="text-4xl font-bold leading-tight text-gray-800 sm:text-5xl lg:text-6xl">
            Play{" "}
            <span className="text-transparent bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text">
              The Crew
            </span>
          </h1>
          <p className="max-w-3xl mx-auto text-lg font-medium leading-relaxed text-gray-600 sm:text-xl">
            Create a new game room and share your URL with your friends!
          </p>
        </div>

        {/* Call to Action */}
        <div className="space-y-4">
          <button
            onClick={handleCreateRoom}
            disabled={isLoading}
            className="px-8 py-4 bg-blue-600 text-white rounded-xl text-lg font-semibold 
                     button-hover disabled:opacity-50 disabled:cursor-not-allowed
                     focus:outline-none focus:ring-4 focus:ring-blue-200
                     w-full sm:w-auto min-w-[280px]"
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg
                  className="w-5 h-5 mr-3 -ml-1 text-white animate-spin"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Creating Room...
              </span>
            ) : (
              "Create New Room"
            )}
          </button>
        </div>
      </div>
    </main>
  );
}
