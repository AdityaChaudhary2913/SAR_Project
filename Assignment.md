# **Take‑Home Assignment: Exploring SAR, ML, and a Simple Geo Web App**

## **1. Goal and spirit of the assignment**

This is an open‑ended assignment to see how you:

- Learn a new technical area (SAR).
- Explore the ML landscape around it (classical, deep, and foundation models).
- Design and implement as much as you reasonably can of a small end‑to‑end system: data → model → simple map‑based UI.

This is **not** about completing a perfect product. It’s about how you think, what you choose to focus on, and how you communicate trade‑offs and unfinished parts.

You can spend **up to 3–4 full days** of effort. If you can’t do everything, that’s completely fine, just explain what you did and what you would do next.

Note : Feel free to use LLMs to write code (recommended infact), but be explicit about the parts written using LLM and what they do exactly.

---

## **2. High‑level outline (flexible)**

Here is a suggested structure; you can modify the steps as per your understanding :

1. Learn the basics of SAR and write a short, informal technical note.
2. Skim various ML approaches on SAR data and pick a small use case you find interesting.
3. Implement as much as you can of an end‑to‑end training + inference pipeline for that use case.
4. Wrap the inference in a minimal web map UI where a user can select an Area of Interest (AOI) and see some output.

You don’t have to follow this exactly, but try to cover all four themes in some way.

---

## **3. Learning SAR and writing a short note**

Spend some time getting comfortable with SAR:

- Read or watch 1–2 good introductions to SAR (for example, NASA ARSET’s “Introduction to Synthetic Aperture Radar (SAR) and its Applications” or similar resources you find).

Then write a brief note (it can be a Markdown file, PDF, or even a well‑structured README section) where you explain in your own words:

- What SAR is and how it’s different from optical imagery.
- A few key concepts that you think matter for ML (e.g., polarization, incidence angle, speckle, basic preprocessing).
- A couple of application areas that seem interesting to you (e.g., flood mapping, land‑cover mapping, urban change).

This doesn’t need to be long or formal, 2–4 pages (or equivalent) is plenty. The goal is to see how you understand and re‑explain new material.

---

## **4. Exploring SAR ML models and picking a use case**

Next, do a lightweight survey of “SAR + ML”:

- Look at a few classical methods (e.g., texture features + SVM/RF) and a few deep learning examples (e.g., SAR segmentation, detection, land cover).
- Notice at least one foundation or pre‑trained model that could be relevant (e.g., a geospatial foundation model or a SAR benchmark/foundation model).

In your note or a separate section, briefly summarize:

- A few models/datasets you found (name, task, type of data, link is enough).
- One **small** use case you would like to try in this assignment, for example:
    - SAR‑based land‑cover classification (urban vs non‑urban, water vs non‑water, etc.).
    - Flood/water detection on SAR.
    - Any other simple task you can support with open SAR data.

Please **avoid just re‑implementing a single online tutorial** as‑is. It’s okay to borrow code or ideas, but put your own spin on the problem (different AOI, slightly different labels, different evaluation, etc.).

---

## **5. Implementing a simple SAR ML pipeline**

Try to build as much as you can of an end‑to‑end pipeline for your chosen use case:

## **Data**

- Use public SAR data (e.g., Sentinel‑1 or an open SAR dataset).
- Keep the scope small so it’s manageable (a city/region or a few scenes is fine).
- Write down:
    - Where you got the data from.
    - Rough area and time period.
    - Any basic preprocessing you applied (e.g., normalization, simple filtering, tiling).

## **Models**

Aim for at least:

- One **simple baseline** model (this could be classical ML on simple features or a very small neural net).
- One **slightly more advanced** model (e.g., a small UNet or CNN, or a pre‑trained encoder plus a small head).

If you run out of time:

- It’s okay if you only get one model working and sketch the other in a design doc.
- It’s okay if training is on a very small subset of data.

## **Code structure**

Organize your code in a way that feels natural to you, but try to keep it clean and modular. For example:

- Separate scripts/modules for:
    - Data loading and preprocessing.
    - Model definition.
    - Training loop.
    - Inference.

Use config files or CLI arguments rather than hard‑coding everything. It doesn’t have to be fancy, just something a stranger can follow.

## **Evaluation**

Do a light evaluation:

- A simple metric suitable for your task (e.g., accuracy, IoU, F1).
- A couple of qualitative examples (e.g., showing predictions overlayed on an image or map, or a before/after comparison).

If you don’t have enough time to fully evaluate, describe what you **would** do and show whatever partial results you have.

---

## **6. Minimal web map UI with AOI selection**

Finally, build a very small web demo. The goal is just to connect the model to a map UI, not to build a full product.

## **Backend (very lightweight)**

- A simple HTTP endpoint (FastAPI/Flask/Django or any framework you like).
- The endpoint can accept:
    - A bounding box / polygon; or
    - A simpler identifier that stands in for a pre‑defined tile (e.g., “tile_1”).

Internally, you can:

- Map the AOI to existing data you have.
- Run your inference (or load a pre‑computed result if running live is too heavy).
- Return something the frontend can visualize (e.g., an image URL, a small GeoJSON, or a simple raster‑like matrix).

## **Frontend**

- A simple web page (any stack you like; React + Leaflet is a common choice) with:
    - A map.
    - A way to select or draw an AOI (even just a draggable rectangle or clicking on the map).
    - A button to “Run inference” that calls your backend and then displays the result somehow (overlay, markers, legend, etc.).

Simplifications are absolutely fine:

- Fixed geographic region.
- Only rectangles, not arbitrary polygons.
- Pre‑computed inference results that you look up instead of computing on the fly.

The aim is to show that you can design the flow AOI → backend → model result → map.

---

## **7. What to submit**

When you’re done (or when you’re out of time), please send:

- A link to your Git repository (GitHub/GitLab/etc.).
- Your SAR/ML notes or report (if not already in the repo).
- Basic instructions on:
    - How to set up the environment.
    - How to run training (if feasible).
    - How to start the backend and the frontend and trigger a sample inference.

It’s perfectly okay to mention parts that are incomplete, hacked together, or left as future work—just call them out clearly.

---

## **8. How I’ll look at your work**

I’m mainly interested in:

- How you learn and explain new concepts.
- How you explore options and pick a reasonable use case.
- How you structure your code and make the system understandable.
- How you deal with practical constraints (time, data, compute) and communicate trade‑offs.

A partially working but well‑thought‑out project with clear explanations is **more valuable** than a “finished” one with copy‑pasted code and no reasoning.

Looking forward to learn and discuss more once you complete it.