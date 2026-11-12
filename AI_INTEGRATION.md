# Connecting the AI project later

The exported frontend currently provides visual design and browser-local preview. Real analysis needs a separate inference backend. No results should be invented to fill the interface.

## Required project inputs

- Validated FUnIE-GAN checkpoint and its preprocessing/postprocessing code.
- Fine-tuned RF-DETR checkpoint, exact model variant, label mapping and inference configuration.
- Tested inference environment with its pinned dependency versions.
- A server able to load these models and process images within an acceptable time.
- Defined report rules, uncertainty handling and real evaluation results.

## Proposed integration contract — not currently implemented

A backend endpoint such as POST /api/analyze can accept multipart image data and return:

- original image dimensions;
- enhanced image content or a short-lived authorized result URL;
- detections containing class label, confidence and xyxy box coordinates in ORIGINAL image pixels;
- image quality / uncertainty notes;
- screening findings, priority and explanation;
- model/checkpoint identifiers for reproducibility.

Implement server-side decoding and size checks as well as the client checks. Define timeouts, failures, retention/deletion, rate limits and any cross-origin configuration. Keep service credentials on the server, never in app.js.

## Frontend work when the backend is ready

1. Enable Analyze only when a valid image and an available service exist.
2. Submit the image with explicit loading/error/cancel states.
3. Render real original/enhanced comparison and coordinate-correct detection boxes.
4. Carry the result to the Analysis Report page with a chosen session/result lifecycle.
5. Generate downloads from actual result data.
6. Update the upload privacy wording to match the implemented retention policy.

The current report page has no results and no permanent storage. User images must not silently become training data. Environmental screening is not proof of plastic polymer type or chemical toxicity. A failed/no detection is not proof that the scene is safe or debris-free.

The Contact page already has mailto links. Its form only downloads a draft; sending through the form would require an additional mail-delivery backend.
