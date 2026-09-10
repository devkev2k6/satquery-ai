You are working in the satquery-ai project folder, received from M3 (VQA & captioning).
This is step 4 of a 6-person pipeline.

FIRST: Read docs/HANDOFF.md in full. Pay attention to M2's answer_question() function
signature and M3's backend/vqa_tool.py and backend/captioning_tool.py — your new tools
should follow the SAME dict-return pattern they established, for consistency. Also check
M1's data/loader.py for load_cdvqa() (bi-temporal change pairs).

YOUR TASK: Build the two multi-image capabilities required by the problem statement:
(a) change detection/description from a bi-temporal (before/after) image pair —
MANDATORY, and (b) joint optical+SAR cross-modal analysis on a co-registered image pair
— MANDATORY. These are two separate tools; build both.

Do the following, in order:

1. Create backend/change_detection_tool.py with a function:
   run_change_detection(image_path_before: str, image_path_after: str, question: str = None) -> dict
   It should:
   - If question is provided, answer it about what changed (change-based VQA)
   - If no question, produce a general change description
   - You can implement this by calling M2's answer_question() twice (once per image) and
     combining the answers into a comparison, OR by prompting more cleverly if the model
     supports multi-image input — document whichever approach you used and why
   Return format:
   {"task": "change_detection", "description": "...", "change_detected": true/false,
    "confidence": 0.0-1.0, "before_path": "...", "after_path": "..."}

2. Create backend/fusion_tool.py with a function:
   run_optical_sar_fusion(optical_path: str, sar_path: str, question: str = None) -> dict
   This must extract complementary information from a co-registered optical and SAR pair
   (e.g. "identify built-up and water-covered regions using both images together"). If a
   true multimodal fusion model is out of scope for the timeline, implement a reasonable
   simplified version: run M2's model separately on each image, then combine the two
   textual outputs into a single synthesized answer, clearly noting in code comments that
   this is a text-level fusion approach rather than pixel-level fusion. Return:
   {"task": "fusion", "answer": "...", "confidence": 0.0-1.0, "optical_path": "...",
    "sar_path": "..."}

3. Write tests/test_change_and_fusion.py testing both tools against sample pairs (use
   M1's data/sample/pairs/ folder, or CDVQA sample data if available) and print results.

4. Append to backend/README.md with the same documentation style M3 used: function
   signatures, return formats, caveats.

5. Update docs/HANDOFF.md by APPENDING:
   - Your two function signatures (M5 needs these exactly)
   - Whether your fusion approach is true multimodal fusion or text-level combination —
     be explicit, this affects how the agent controller should describe results to users
   - Known weaknesses (e.g. "change detection sometimes flags lighting differences as
     change")

6. Commit your work to git.

Do NOT modify data/, models/, or M3's backend/vqa_tool.py and captioning_tool.py — only
ADD your two new files to backend/. Do NOT build the agent controller (M5) or frontend (M6).

When finished, run tests/test_change_and_fusion.py and paste the output.
