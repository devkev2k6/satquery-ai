You are working in the satquery-ai project folder, which you're receiving from a
teammate (M1) who set up the datasets. This is step 2 of a 6-person pipeline.

FIRST: Read docs/HANDOFF.md in full — it tells you what M1 built, especially how to use
the functions in data/loader.py. Also read data/README.md. Do not proceed until you
understand what data is available and in what format. If data/sample/ contains synthetic
placeholder data (because real datasets weren't downloaded yet), that's fine — use it to
build and test your pipeline; it should work identically once real data is swapped in.

YOUR TASK: Fine-tune (or lightly adapt) a small open-source vision-language model on the
satellite imagery data, so it can answer basic questions about satellite images. This
satisfies the problem statement's MANDATORY requirement that at least one visual or
vision-language component be adapted to remote-sensing imagery — this is not optional
and is the most heavily weighted requirement, so get a working version even if accuracy
is low.

Do the following, in order:

1. Choose ONE small, open-source, pretrained vision-language model that is realistic to
   fine-tune on a laptop or a single free-tier GPU (e.g. Colab). Prefer something in the
   1B-3B parameter range with an existing image-captioning or VQA head, available via
   Hugging Face transformers. Document your choice and reasoning in models/MODEL_CHOICE.md.

2. Write models/finetune.py that:
   - Loads the base model and processor
   - Loads training data using data.loader.load_bigearthnet() (or load_sample_data() as
     fallback) from M1's code
   - Applies a LIGHTWEIGHT adaptation method (LoRA or similar parameter-efficient
     fine-tuning — NOT full fine-tuning, which is unrealistic for a hackathon timeline)
   - Trains for a small number of steps/epochs suitable for demonstrating adaptation, not
     achieving state-of-the-art accuracy
   - Saves the adapted model weights to models/checkpoints/

3. Write models/inference.py exposing ONE simple, clean function:
   answer_question(image_path: str, question: str) -> str
   This function should load the fine-tuned model and return a text answer. This exact
   function signature matters — teammates after you (M3, M4, M5) will import and call it
   directly, so do not change the name or arguments without updating docs/HANDOFF.md.

4. Write a small evaluation script models/evaluate.py that runs the model against a
   handful of VRSBench and RSVQA sample questions (from M1's loaders) and prints
   question/expected-answer/model-answer side by side, so the team can sanity-check
   quality at a glance.

5. Write models/README.md documenting: the base model used, the adaptation method, how to
   re-run training, how to call answer_question(), expected inference time, and current
   known limitations/failure modes (be honest — e.g. "struggles with counting objects" is
   useful information for M3).

6. Update docs/HANDOFF.md by APPENDING:
   - What model you used and why
   - Confirmation that answer_question(image_path, question) works and how to call it
   - Any GPU/hardware requirements the next teammates need to know about
   - Recommendation on whether M3-M4 should call your model directly for their tasks, or
     build lighter rule-based logic for anything your model handles poorly

7. Commit your work to git.

Do NOT modify data/ (M1's work) except to read from it. Do NOT build the backend, frontend,
or agent controller — that's for teammates after you. Your deliverable is a working,
importable answer_question() function plus the training pipeline that produced it.

When finished, run models/evaluate.py and paste its output so we can confirm the model
is actually answering questions (even if answers are imperfect).