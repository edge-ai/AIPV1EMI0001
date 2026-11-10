# Design system

## Direction

The AIPV1EMI0001 dashboard is an operations console, not a reporting portal.
Its first viewport keeps live video and current counts visible together so an
operator can confirm pipeline behavior without switching context.

Use the EdgeAI Slidev theme and its canonical brand kit as visual authority,
not an independently invented dashboard palette. The pinned source is
`edge-ai/aibox-eais-slidev` revision
`a1f7a2fc553d645a0692c2b3b1e4379a91f0fa44`, particularly
`theme/styles/tokens.css` and `brand/generated/css/brand.css`, derived from the
Edgematrix Visual Identity Guidelines (2025-02-07).

Operators enter from the light station web UI, so the surrounding dashboard
stays light; only the video letterbox is dark. No gradients, glows, glass
surfaces, or decorative class colours are used.

## Color

- `#eff2f7`: application page background
- `#ffffff`: panel surface
- `#f5f9ff`: Blue-50 secondary surface
- `#dee2e6`: dividers
- `#343a40`: headings and counts
- `#495057`: secondary text and labels
- `#3b82f6`: Blue-500 focus and activity indicators
- `#326fd1`: Blue-600 button background
- `#1d4ed8`: active blue and button hover
- `#22c55e`, `#f59e0b`, `#ef4444`: success, warning, and error markers
- `#111827` / `#f9fafb`: video background / overlay text

Accessibility adaptations: use the brand's body colour instead of its faint
`#9e9e9e` label colour for small text; `#6b7280` provides a visible control border.
Small error text uses darker `#b91c1c`, and white button text sits on Blue-600
rather than Blue-500. Status markers retain the canonical brand colours but
always have a readable text label; colour alone never communicates state.

## Typography

Inter 4.1 is bundled locally with its SIL Open Font License, using one variable
WOFF2 file and `font-display: swap`. No external font service is permitted.
The fallback is the operating system sans-serif stack.

Use the brand's 14px body/card-title and 17.5px page-title scale, expressed in
rem, with 12px helper text. Changing measurements use tabular numerals, with
the all-object total at 40px. Weights are 400, 600, and 700. Monospace is not
used as a visual shorthand for technical content.

## Layout

Desktop uses a dominant 16:9 video panel with a narrower count console beside
it. ROI and line-crossing details form a two-column row below. At 800px and
below, these regions stack into one column without changing the reading order.

## Components and states

- Panels and controls use the Slidev theme's 4px small radius.
- White panels use a subtle offset shadow; data rows use dividers, not nested cards.
- Compact text-and-dot indicators communicate video waiting, loading, ready,
  playing, disabled, disconnected, and error, without glows or uppercase pills.
- The player follows pipeline output settings; no HLS/WebRTC selector is shown.
- Video errors expose a retry action; disabled outputs show a placeholder.
- The source selector uses the browser's standard control behavior and a
  visible keyboard focus ring.
- Empty states explain what data or pipeline configuration is missing.
- Motion is limited to the waiting-state pulse and respects reduced-motion.

## Offline delivery

The dashboard has no build step or runtime dependency on the Slidev package.
Token values live in `index.css`; the font and license live in `src/fonts/`.
Every new runtime asset must be listed explicitly in `.include` so the AIP
package remains self-contained. Browser libraries are served by the station's
local UIBuilder installation. Do not add CDN imports, remote fonts, or
external image requests.
