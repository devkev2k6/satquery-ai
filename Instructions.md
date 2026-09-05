You are working in the satquery-ai project folder. This is the FIRST step in a 6-person
pipeline — nothing has been built yet except the folder skeleton. Read docs/HANDOFF.md
first (it should just have a placeholder) to confirm you're starting fresh.

YOUR TASK: Set up the datasets this project needs, using small manageable samples (not
full multi-GB downloads) so five more teammates can build on top of your work without
running out of disk space or time.

Do the following, in order:

1. Create a Python virtual environment setup script (scripts/setup_env.sh or setup_env.bat
   for both Linux/Mac and Windows) that installs: numpy, pandas, pillow, rasterio (for
   GeoTIFF handling), requests, tqdm, scikit-learn, matplotlib. Add these to requirements.txt.

2. Download or prepare SMALL sample subsets (aim for under 500MB total, a few hundred
   images per dataset, not the full datasets) of:
   - BigEarthNet (primary dataset for image-text adaptation — this is mandatory per the
     problem statement)
   - VRSBench (for captioning, grounding, VQA evaluation)
   - RSVQA (for single-image VQA evaluation)
   - CDVQA (for multitemporal change-based VQA evaluation)
   If direct download links require registration or are large, write a clear
   data/DOWNLOAD_INSTRUCTIONS.md explaining exactly how a teammate can get the sample data
   themselves (source URLs, expected folder structure, file sizes), and additionally
   generate a small set of SYNTHETIC placeholder images (using PIL, random noise textures
   labeled as "optical" and "sar" style) in data/sample/ so the pipeline can be built and
   tested end-to-end even before real data is downloaded.

3. Organize everything under data/ using this structure:
   data/
     bigearthnet/{images, labels.csv}
     vrsbench/{images, annotations.json}
     rsvqa/{images, qa_pairs.json}
     cdvqa/{image_pairs/, qa_pairs.json}
     sample/{optical/, sar/, pairs/}  (your synthetic fallback data)

4. Write a data-loading utility at data/loader.py with simple, well-documented Python
   functions: load_bigearthnet(), load_vrsbench(), load_rsvqa(), load_cdvqa(), and
   load_sample_data(). Each should return a simple list of dicts like
   {"image_path": ..., "question": ..., "answer": ...} or similar, so later teammates
   don't need to know dataset-specific formats.

5. Write data/README.md explaining: what each dataset is for, how to get the full version
   later, what format loader.py returns, and any known limitations of the sample data.

6. Update docs/HANDOFF.md by APPENDING (do not delete the placeholder, just add below it):
   - What you built and where it lives
   - How to run your setup script
   - Exactly how the next person (M2, doing model fine-tuning) should call your loader
     functions to get training data
   - Any blockers or things you couldn't finish (e.g. "real BigEarthNet download requires
     manual registration, synthetic data is the fallback until then")

7. Commit your work to git with a clear commit message.

Do NOT touch the models/, backend/, frontend/, or agent/ folders — those belong to
teammates after you. Keep your work self-contained in data/ and scripts/.

When finished, print a summary of every file you created and confirm the loader
functions work by running a quick test that loads and prints one sample from each dataset.