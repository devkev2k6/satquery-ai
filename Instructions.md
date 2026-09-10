You are working in the satquery-ai project folder, received from M5 (agent controller).
This is the FINAL step of a 6-person pipeline. Everything else is built — your job is to
make it usable and presentable.

FIRST: Read docs/HANDOFF.md in full, in particular M5's section on
agent.controller.process_query(query, image_paths, image_types). Read agent/controller.py
directly to confirm the exact signature and return format before writing any UI code.
Also run tests/test_agent_controller.py yourself once to confirm the backend actually
works before building on top of it — do not assume, verify.

YOUR TASK: Build a simple, clean web application that lets a user upload 1-2 images,
type a natural-language question, and see the result — text answer, execution summary,
and visual evidence (the uploaded image, with a highlighted region if grounding was used).
Also produce the downloadable PDF report feature and prepare demo materials.

Do the following, in order:

1. Build frontend/app.py using Streamlit (fastest realistic option for a hackathon demo;
   use Flask + a simple HTML template instead ONLY if the team specifically prefers that —
   otherwise default to Streamlit) with:
   - A file uploader supporting 1 or 2 images (GeoTIFF/TIFF, plus PNG/JPEG for benchmark
     datasets)
   - A dropdown or radio buttons to optionally tag each uploaded image as "optical" or
     "sar" (needed for the fusion task to be routed correctly)
   - A text input box for the natural-language query
   - A "Submit" button that calls agent.controller.process_query() with the uploaded
     image paths and query
   - A results panel showing: the text answer/description prominently, the execution
     summary (task selected, tool used, parameters) in a collapsible/expandable section
     so it doesn't clutter the main view, and the uploaded image(s) displayed inline
   - Basic error handling: if process_query() returns success: false, show the error
     message clearly instead of crashing

2. Write frontend/report_generator.py with a function:
   generate_pdf_report(query: str, result: dict) -> str (returns path to saved PDF)
   That creates a simple PDF (use fpdf2 or reportlab) summarizing: the query asked, the
   task the agent selected, the tool used, the answer/result, and a timestamp. Wire a
   "Download Report" button in app.py to this function.

3. Write tests/test_full_demo.py that runs all 5 representative queries from the problem
   statement through the SAME path the UI uses (i.e. call process_query() the way app.py
   does) and confirms each produces a valid result and a downloadable report — this is
   your final end-to-end sanity check before the live demo.

4. Write the top-level README.md (replace the placeholder) with: project overview, setup
   instructions (clone, install requirements.txt, run app.py), a screenshot or description
   of the UI, and credit to each team member's contribution area.

5. Update docs/HANDOFF.md with a final entry: confirm the full pipeline works end-to-end,
   list any known bugs or rough edges for the team to be aware of during the live demo,
   and note anything that should be mentioned as a "future work" limitation if judges ask
   about accuracy or scale.

6. Commit your work to git with a final commit.

Do NOT modify backend/, models/, agent/, or data/ except to import/call from them — if
something is broken there, note it in docs/HANDOFF.md rather than silently patching
someone else's module, so the team knows to fix it together before the demo.

When finished:
- Run the Streamlit app locally and confirm it works for all 5 representative queries
- Run tests/test_full_demo.py and paste the output
- List every file in the final project structure so the whole team can review it before
  submission