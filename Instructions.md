You are working in the satquery-ai project folder, received from M4 (change detection &
fusion). This is step 5 of a 6-person pipeline, and arguably the most important step for
scoring well — the problem statement explicitly evaluates the agentic orchestration
(task selection, tool execution, output integration, and an auditable execution summary).

FIRST: Read docs/HANDOFF.md in full. You now have FOUR tools available, built by M3 and
M4, all following the same dict-return pattern:
  - backend/vqa_tool.py → run_vqa(image_path, question)
  - backend/captioning_tool.py → run_captioning(image_path)
  - backend/grounding_tool.py → run_grounding(image_path, query)   [if M3 built it]
  - backend/change_detection_tool.py → run_change_detection(before, after, question)
  - backend/fusion_tool.py → run_optical_sar_fusion(optical, sar, question)
Read each tool file directly to confirm exact signatures before writing code — the
handoff notes summarize them but the source files are the ground truth.

YOUR TASK: Build the agentic controller. This is the "brain" that: interprets a user's
natural-language query, checks how many/what type of images were provided, decides which
tool(s) above to call, executes them, and returns a combined, evidence-grounded response
with a transparent record of what it did. This satisfies the problem statement's
"Agentic Model and Tool Orchestration" section directly — read that section's exact
requirements again before starting.

Do the following, in order:

1. Create agent/task_classifier.py with a function:
   classify_task(query: str, num_images: int, image_types: list) -> str
   That returns one of: "vqa", "captioning", "grounding", "change_detection", "fusion".
   Start with a clear, well-commented RULE-BASED classifier (keyword matching: words like
   "changed"/"between"/"before and after" → change_detection; "highlight"/"where is" →
   grounding; two images where one is optical and one is SAR → fusion; single image with
   a question → vqa; single image, no specific question or "describe" → captioning).
   This is intentionally simple and explainable — a rule-based classifier still fully
   satisfies the orchestration requirement, and it's easier to defend to judges than an
   opaque model. Only add an LLM-based fallback classifier if time allows, keep the
   rule-based version as the default.

2. Create agent/input_validator.py with a function:
   validate_input(query: str, image_paths: list, image_types: list) -> dict
   That checks: correct number of images for the requested task (e.g. reject
   change_detection with only 1 image), supported file formats, and that image types are
   compatible with the task (e.g. fusion needs one optical + one SAR). Return
   {"valid": true/false, "error": "..." or None}.

3. Create agent/controller.py — the main orchestrator — with a function:
   process_query(query: str, image_paths: list, image_types: list = None) -> dict
   That:
   a. Calls classify_task() to determine the task
   b. Calls validate_input() and returns a clear error dict if invalid
   c. Calls the correct backend tool function with the correct arguments
   d. Wraps the tool's result together with an EXECUTION SUMMARY — this is explicitly
      required by the problem statement. The summary must include: selected task, which
      tool/model was called, key parameters used, and a timestamp. Return format:
      {"result": {...tool output...},
       "execution_summary": {"task": "...", "tool_used": "...", "parameters": {...},
                              "timestamp": "..."},
       "success": true/false}

4. Write tests/test_agent_controller.py that runs process_query() against ALL FIVE
   representative example queries from the original problem statement:
   - "Describe the land-cover and major objects visible in this image."
   - "Highlight the water body referred to in the query."
   - "What changed between these two dates, and where did the change occur?"
   - "Use the optical and SAR images together to identify built-up and water-covered
     regions."
   - "Has the built-up area increased, decreased, or remained unchanged?"
   Print the full result (including execution_summary) for each so the team can visually
   confirm correct task routing before the demo.

5. Write agent/README.md documenting the controller's decision logic, the exact rules
   used in classify_task(), and how to extend it with a new task type in the future.

6. Update docs/HANDOFF.md by APPENDING:
   - Confirmation that process_query(query, image_paths, image_types) works end-to-end
   - Exact function signature M6 (frontend) needs to call
   - Any known misclassification cases (e.g. "ambiguous queries mentioning both 'compare'
     and 'highlight' may misroute — flag this as a known limitation in the demo")

7. Commit your work to git.

Do NOT modify backend/ tool files or models/ — only READ from them. Do NOT build the
frontend (M6).

When finished, run tests/test_agent_controller.py and paste the full output for all 5
example queries — this is the core proof the system works end to end.
