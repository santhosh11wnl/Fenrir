/**
 * The floating launcher.
 *
 * A circle that lives in the corner of the host site and morphs into the chat
 * panel. Details that matter more than they look:
 *
 * - It is a real `<button>` with an accessible name, so it is keyboard
 *   reachable and announced. A styled `<div>` would be invisible to a screen
 *   reader and unreachable by Tab.
 * - The icon crossfades between chat and close rather than swapping, so the
 *   control reads as one object changing state rather than two buttons.
 * - The idle ring animation stops while the panel is open: an attention cue
 *   that keeps pulsing after it has been answered is just noise.
 */

interface LauncherProps {
  open: boolean;
  unread: boolean;
  busy: boolean;
  onClick: () => void;
}

export function Launcher({ open, unread, busy, onClick }: LauncherProps) {
  return (
    <button
      type="button"
      className={`launcher${open ? " launcher--open" : ""}${busy ? " launcher--busy" : ""}`}
      onClick={onClick}
      aria-label={open ? "Close chat" : "Open chat"}
      aria-expanded={open}
    >
      {/* Concentric rings that breathe outward while idle. Purely decorative,
          so hidden from assistive tech. */}
      <span className="launcher__halo" aria-hidden="true" />
      <span className="launcher__halo launcher__halo--delayed" aria-hidden="true" />

      <span className="launcher__face" aria-hidden="true">
        <svg className="launcher__icon launcher__icon--chat" viewBox="0 0 24 24" fill="none">
          <path
            d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>

        <svg className="launcher__icon launcher__icon--close" viewBox="0 0 24 24" fill="none">
          <path
            d="M18 6 6 18M6 6l12 12"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
          />
        </svg>
      </span>

      {/* Only meaningful when the panel is closed -- an unread dot over an
          open conversation would be telling you about something you can see. */}
      {unread && !open && <span className="launcher__badge" aria-hidden="true" />}
    </button>
  );
}
