import type { Copy } from "../i18n";
import type { Waifu2xNoise, Waifu2xScale, Waifu2xSettings } from "../waifu2x";

interface SegmentedProps<T extends string | number> {
  label: string;
  options: readonly T[];
  value: T;
  render: (option: T) => string;
  onChange: (option: T) => void;
}

function Segmented<T extends string | number>({ label, options, value, render, onChange }: SegmentedProps<T>) {
  return (
    <div>
      <p className="mb-1.5 text-xs font-medium text-[var(--ink-muted)]">{label}</p>
      <div
        className="grid gap-1 rounded-full border border-[var(--line)] bg-[var(--paper)] p-1"
        style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
        role="radiogroup"
        aria-label={label}
      >
        {options.map((option) => (
          <button
            key={option}
            type="button"
            role="radio"
            aria-checked={value === option}
            onClick={() => onChange(option)}
            className={`min-h-8 rounded-full px-2 text-xs font-medium transition-colors active:scale-[0.98] ${
              value === option
                ? "bg-[var(--ink)] text-[var(--paper)]"
                : "text-[var(--ink-muted)] hover:text-[var(--ink)]"
            }`}
          >
            {render(option)}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Waifu2xControls({
  settings,
  onChange,
  t,
}: {
  settings: Waifu2xSettings;
  onChange: (settings: Waifu2xSettings) => void;
  t: Copy;
}) {
  // 1x without noise reduction is a no-op the backend rejects; nudge the
  // other control instead of letting the user build it.
  const setScale = (scale: Waifu2xScale) =>
    onChange({ ...settings, scale, noise: scale === 1 && settings.noise < 0 ? 1 : settings.noise });
  const setNoise = (noise: Waifu2xNoise) =>
    onChange({ ...settings, noise, scale: noise < 0 && settings.scale === 1 ? 2 : settings.scale });

  return (
    <div className="mt-3 space-y-3 rounded-2xl border border-[var(--line)] p-3">
      <Segmented
        label={t.waifu2xModel}
        options={["art", "art_scan", "photo"] as const}
        value={settings.model}
        render={(model) => t.waifu2xModels[model]}
        onChange={(model) => onChange({ ...settings, model })}
      />
      <Segmented
        label={t.waifu2xScale}
        options={[1, 2, 4] as const}
        value={settings.scale}
        render={(scale) => `${scale}×`}
        onChange={setScale}
      />
      <Segmented
        label={t.waifu2xNoise}
        options={[-1, 0, 1, 2, 3] as const}
        value={settings.noise}
        render={(noise) => (noise < 0 ? t.waifu2xNoiseOff : String(noise))}
        onChange={setNoise}
      />
      <p className="text-xs leading-5 text-[var(--ink-muted)]">
        {t.waifu2xModelHints[settings.model]} {settings.scale === 4 ? t.waifu2xScale4Hint : ""}
      </p>
    </div>
  );
}
