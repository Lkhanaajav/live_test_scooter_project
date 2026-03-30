# Stack Research — Thesis Writing Tools & Practices

## LaTeX Best Practices for Thesis Writing

### Package Recommendations
- **booktabs** — professional table formatting (already in use; keep `\toprule`, `\midrule`, `\bottomrule`)
- **siunitx** — consistent unit formatting (already in use; use `\SI{}{}` for all measurements)
- **subcaption** — subfigures (already in use)
- **cleveref** — smart cross-references (`\cref{fig:x}` auto-generates "Figure 1" consistently)
- **microtype** — micro-typographic refinement (already enabled; improves line breaks)
- **threeparttable** — table footnotes (already in use)
- **algorithm2e** or **algpseudocode** — pseudocode (already using algpseudocode)

### Bibliography Management
- Use `natbib` with `unsrtnat` (already configured) — numbered references sorted by appearance
- Ensure every `\cite{}` corresponds to a real entry in `references.bib`
- For web references (Starship, Ultralytics), include `url` and `note = {Accessed: date}` fields
- Group related citations: `\cite{segformer,deeplabv3plus,bisenetv2}` not three separate `\cite{}`

### Figure Quality Standards
- All figures should be vector (PDF) or high-resolution raster (>300 DPI PNG)
- Consistent font sizes across all figures (match body text ~10-11pt)
- Axis labels should be readable without zooming
- Color schemes should be colorblind-friendly (avoid red/green only)
- Every figure must be referenced in the text *before* it appears

## Academic Writing Style for Engineering Theses

### Voice and Tone
- Use **first-person plural** ("we propose", "we evaluate") — standard in engineering theses
- Active voice preferred: "We compare five methods" not "Five methods were compared"
- Present tense for established facts and the system description
- Past tense for what was done experimentally
- Avoid hedging language: "The results show" not "The results seem to suggest"

### Paragraph Structure
- Lead sentence states the claim
- Supporting sentences provide evidence
- Final sentence connects to the next paragraph or draws a conclusion
- One idea per paragraph; one paragraph per idea

### Quantitative Writing
- Always report units: `14.3 px` not just `14.3`
- Report improvements as both absolute and relative: "from 0.758 to 0.946 (24.8% improvement)"
- Use consistent decimal places within a table
- Bold the best result in comparison tables
- Report confidence intervals or standard deviation when possible

## Evaluation Presentation Standards

### For Computer Vision Papers/Theses
- **Always compare against a baseline** — not just ablations of your own system
- Standard baselines: (1) stock pretrained model without fine-tuning, (2) simplest reasonable approach, (3) published methods if available
- Report IoU, Precision, Recall, F1 as a group — not just IoU
- For runtime: report hardware specs, batch size, input resolution, whether GPU/CPU

### For Robotics/Planning Papers/Theses
- Report both accuracy AND latency — neither alone is sufficient
- Include failure analysis: when does the system break?
- Runtime should be per-frame wall clock, not FLOPS
- Include qualitative examples for both success and failure cases

### Table Design
- Minimize visual clutter: no vertical lines, use `booktabs` rules only
- Group related rows with `\midrule` or indent
- Caption above the table (standard for IEEE-style)
- Caption should be self-contained — reader should understand the table without reading body text

## Proofreading and Revision Workflow

1. **Structural pass** — outline-level review of chapter flow and argument arc
2. **Content pass** — verify every claim has evidence, every figure is referenced
3. **Consistency pass** — terminology, notation, abbreviation usage
4. **Style pass** — sentence-level clarity, passive→active voice, paragraph transitions
5. **Format pass** — page numbers, margins, TOC, list of figures/tables
6. **Final read** — print or PDF read-through for typos and formatting glitches
