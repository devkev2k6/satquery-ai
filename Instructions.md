You are working in the satquery-ai project folder, received from M2 (model fine-tuning).
This is step 3 of a 6-person pipeline.

FIRST: Read docs/HANDOFF.md in full, paying close attention to the section M2 wrote about
their answer_question(image_path, question) function in models/inference.py — you will
build directly on top of it. Also skim models/README.md for known model limitations.

YOUR TASK: Build the single-image baseline features required by the problem statement:
(a) Visual Question Answering on a single image — MANDATORY, and (b) at least one of
captioning/scene description OR text-guided region grounding — pick ONE, build it well.
Recommend picking CAPTIONING first since it's simpler to demo reliably; only attempt
grounding (drawing a box around a described object) if time allows.

Do the following, in order:

1. Create backend/vqa_tool.py with a function:
   run_vqa(image_path: str, question: str) -> dict
   It should call M2's answer_question() from models/inference.py, and return a
   structured dict like:
   {"task": "vqa", "answer": "...", "confidence": 0.0-1.0, "image_path": "..."}
   Confidence can be a simple heuristic for now (e.g. based on answer length/certainty
   phrasing) — document however you calculate it.

2. Create backend/captioning_tool.py with a function:
   run_captioning(image_path: str) -> dict
   This should produce a natural-language description of the image's land cover and
   major visible objects. You can implement this by prompting M2's model with a fixed
   captioning-style question (e.g. "Describe the land cover and major objects visible in
   this image.") if the model doesn't have a dedicated captioning mode. Return:
   {"task": "captioning", "caption": "...", "confidence": 0.0-1.0, "image_path": "..."}

3. IF TIME ALLOWS, create backend/grounding_tool.py with a function:
   run_grounding(image_path: str, query: str) -> dict
   That attempts to identify and return approximate bounding-box coordinates for the
   object described in query (e.g. "the water body"). If the underlying model can't do
   this reliably, implement a clearly-labeled simplified version (e.g. basic color/texture
   heuristics for water/vegetation) and document this limitation clearly — do not claim
   accuracy you don't have. Return:
   {"task": "grounding", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0, "image_path": "..."}

4. Write tests/test_vqa_captioning.py that runs both tools against 3-5 sample images from
   M1's data/sample/ folder and prints results, so anyone can verify the tools work by
   running one command.

5. Write backend/README.md (create the backend/ README if it doesn't exist yet — future
   teammates will keep adding to this same file) documenting each function's exact
   signature, return format, and any caveats.

6. Update docs/HANDOFF.md by APPENDING:
   - Which functions you built and their exact signatures (this matters a lot — M5 the
     agent lead will call these directly)
   - Whether you implemented captioning, grounding, or both
   - Known weaknesses or edge cases (e.g. "captions are generic for cluttered urban
     scenes")

7. Commit your work to git.

Do NOT modify data/ or models/ except to import/call from them. Do NOT build change
detection (M4), the fusion tool (M4), the agent controller (M5), or the frontend (M6).

When finished, run tests/test_vqa_captioning.py and paste the output.
