# Design system

## Direction

The AIPV1EMI0001 dashboard is an operations console, not a reporting portal.
Its first viewport keeps live video and current counts visible together so an
operator can confirm pipeline behavior without switching context.

## Color

- `#07111f`: page background
- `#101d2d`: primary panel surface
- `#15263a`: raised panel surface
- `#263b52`: borders and separators
- `#edf6ff`: primary text
- `#8ea5ba`: secondary text
- `#35d5e7`: live state and primary telemetry emphasis
- `#55db9c`: healthy/connected and person count marker
- `#f5b942`: warning and truck count marker
- `#ff6b6b`: disconnected/error

Cyan is reserved for live state and important telemetry. It is not a general
decorative accent.

## Typography

Use the operating system sans-serif stack for reliable offline rendering.
Headings are compact and fixed-size; changing measurements use tabular numerals.
Monospace is not used as a visual shorthand for technical content.

## Layout

Desktop uses a dominant 16:9 video panel with a narrower count console beside
it. ROI and line-crossing details form a two-column row below. At 800px and
below, these regions stack into one column without changing the reading order.

## Components and states

- Panels use one radius, border, and surface treatment.
- Status pills communicate HLS loading, ready, playing, unsupported, and error.
- The source selector uses the browser's standard control behavior.
- Empty states explain what data or pipeline configuration is missing.
- Motion is limited to the waiting-state pulse and respects reduced-motion.
