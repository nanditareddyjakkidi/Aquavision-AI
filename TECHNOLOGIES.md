# Technologies used

## Shipped website

| Technology | Actual use |
| --- | --- |
| HTML5 | Five pages, forms, accessible navigation, native video/audio elements |
| CSS3 | Ocean palette, layout, media queries, translucent panels, smooth crossfades |
| Vanilla JavaScript | DOM events, scrolling/depth calculation, video control, sound toggle, menu and file preview |
| Browser File / Blob / URL APIs | Preview selected images locally and download contact message drafts |
| MP4 / H.264 | Locally served real underwater video clips |
| JPEG | Video poster frames |
| MP3 | Optional ocean-wave ambience |
| SVG | Small favicon |
| Google Fonts | DM Sans and Manrope, with system-font fallback |

This website does not use React, Next.js, Tailwind, Bootstrap, a database or an authentication framework. There is no JavaScript package build step.

## Tools used while creating it

- ChatGPT/Codex: code authoring, content/layout changes and packaging.
- Python: file processing and this portable local-preview server.
- FFmpeg: trimming, reframing and compressing videos, extracting posters, converting sound.
- Node.js: JavaScript syntax checks and the hosted publishing workflow. Not required for local use of this export.
- Git: source version history for the hosted site. The ZIP is a clean source snapshot.
- ChatGPT Sites: publishing and website hosting; its deployment tools returned a Cloudflare-hosted static site. No Sites-specific runtime is needed for local use.

## AI models planned for the larger project

1. FUnIE-GAN — underwater image enhancement.
2. RF-DETR Small — debris detection after fine-tuning on the audited project labels.
3. A separate rule-based screening component — review priority and environmental concerns; not a trained toxicity model.

These are planned integration components, not libraries running in the current website. Their frameworks, versions and hardware requirements must be recorded from the actual training/inference environment when it is connected.
