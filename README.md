# AquaVision — Website source package

Exported 30 September 2026 from published website version 8.
Live website: https://aquavision-ocean.jakkidinandita.chatgpt.site
Source revision: 2b44df24e9546d89d5b32f53934a76e3eb3e0cf2

## Save it in your AquaVision project

1. Extract this ZIP. It creates an `AquaVision_Website` folder.
2. Move that entire folder inside your existing `AquaVision` project folder. Keep your existing datasets, training code and model files where they are.
3. Open `AquaVision_Website` in a code editor such as VS Code.
4. Open a terminal in that folder.

### Run on macOS or Linux

```sh
python3 serve.py
```

### Run on Windows

```powershell
py serve.py
```

Open **http://localhost:8000** in your browser. Stop with Ctrl+C.
If port 8000 is busy, run `python3 serve.py --port 8001` (or `py serve.py --port 8001` on Windows), then visit http://localhost:8001.
Python 3 must be installed. No pip, npm, node_modules, API key or database is needed to run this frontend.

Do not double-click index.html: the website uses root-relative paths and needs a local web server. The server must serve the `website` folder as its root. The included script handles this.

## Where to edit

| File | Purpose |
| --- | --- |
| website/index.html | Opening page: ocean education, videos, depth journey |
| website/detection/index.html | Image selection and preview |
| website/report/index.html | Report empty state and explanation |
| website/about/index.html | Project workflow |
| website/contact/index.html | Team email links and downloadable message draft |
| website/style.css | Colors, typography, responsive layout, transitions |
| website/app.js | Navigation, depth meter, video/audio controls, upload preview, message download |
| website/assets/ | Current videos, posters and ocean sound |
| website/favicon.svg | Browser-tab icon |
| serve.py | Local development server only |
| TECHNOLOGIES.md | Stack and current capability summary |
| AI_INTEGRATION.md | What still needs building for real inference |
| MEDIA_SOURCES.md | Media provenance and license links |

Edit a file, save it, then refresh your browser. Local edits do not update the hosted website automatically.

## What is working

- Five navigable pages, no application login.
- Six actual underwater video scenes with depth-based transitions.
- Pause/play motion and reduced-motion support.
- Optional ocean-wave sound, off by default.
- JPG/PNG/WebP validation and local image preview (15 MB limit).
- Clickable team email links.
- Contact form downloads a text draft; it does not send messages.

## What is not connected

The package does NOT contain FUnIE-GAN or RF-DETR inference code, trained checkpoints, public datasets, an API server, generated detections or report generation. The Analyze button is disabled intentionally. This is the complete current website frontend, not the completed AI system.

Your images remain in the browser preview in this version. There is no upload server or permanent image storage. The local server serves files; it does not run the AI models.

## Internet and privacy

All used ocean videos, posters and audio are included locally. Google Fonts are loaded from the internet; without internet the browser uses fallback fonts. Source links and mailto links also depend on external services or your configured email application.

Contact emails are intentionally public as requested. This export excludes credentials, deployment tokens, Git history, unused original media and model/dataset files.

## Checks before sharing changes

Open all five pages. Check the menu on a narrow screen. Scroll through the depth zones. Toggle sound and motion. Preview a supported image and remove it. Confirm invalid/oversized files are rejected. Check that the contact form downloads a draft and does not claim to send it.
