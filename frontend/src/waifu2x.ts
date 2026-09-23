export type Waifu2xModel = "art" | "art_scan" | "photo";
export type Waifu2xScale = 1 | 2 | 4;
export type Waifu2xNoise = -1 | 0 | 1 | 2 | 3;

export interface Waifu2xSettings {
  model: Waifu2xModel;
  scale: Waifu2xScale;
  noise: Waifu2xNoise;
}

// Mirrors Waifu2xSettings() in opennomark/enhancer.py.
export const defaultWaifu2xSettings: Waifu2xSettings = { model: "art", scale: 2, noise: 1 };
