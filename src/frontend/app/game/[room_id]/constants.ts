export const SUIT_COLORS = {
  0: "bg-[#3498db]", // blue
  1: "bg-[#2ecc71]", // green
  2: "bg-[#ff69b4]", // pink
  3: "bg-[#f1c40f]", // yellow
  4: "bg-[#1a2530]", // trump
} as const;

export const PHASE_COLORS = {
  draft: "bg-[#2ecc71]", // green
  signal: "bg-[#f1c40f]", // yellow
  play: "bg-[#3498db]", // blue
} as const;