"""Shared constants for training and for the routing, buffer and
demographic-marking scripts.

Used by train_cl.py, train_prompts_cl.py, routing_analysis.py,
plot_routing_heatmaps.py, buffer_analysis.py and mark_missing_demographics.py.
Edit values here; the scripts only import them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tasks and datasets
# ---------------------------------------------------------------------------
# Task folder names (in image paths) in task order -> task 1..11
TASK_NAMES = [
    "ff++",
    "dfd",
    "celebdf",
    "dfdc",
    "VQGAN",
    "STYLEGAN",
    "Palette",
    "StyleGAN2",
    "DCFACe",
    "stylegan3",
    "StableDiffusion",
]
# Split folder that follows the task folder: <task>/<split>/<real|fake>/<file>
SPLITS = r"(?:train|test|val|valid|validation)"
# Class slices analysed separately (besides all samples)
CLASSES = ("real", "fake")
# Integer label codes, if a label column holds numbers (IndxImageFolder order)
LABEL_CODES = {"0": "fake", "1": "real"}
# Sources whose (real) images have no demographic annotation
UNANNOTATED_SOURCES = ["casia", "celeba", "metfaces"]

# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------
# Demographic annotation CSV per split; each covers all tasks
METADATA_CSV = {
    "train": "metadata/train.csv",
    "val": "metadata/val.csv",
    "test": "metadata/test.csv",
}
# Columns attached to the training datasets (and written to prediction records)
DEMO_ATTRS = [
    "Predicted Gender",
    "Predicted Age",
    "Skin Tone",
    "Skin Tone Group",
    "Intersection",
]
# Attributes analysed (Skin Tone Group dropped); names as in metadata/predictions
DEMOGRAPHICS = [
    "Predicted Gender",
    "Predicted Age",
    "Skin Tone",
    "Intersection",
]
# Display names used in tables and figures. Edit to match the paper.
DEMOGRAPHIC_LABELS = {
    "Predicted Gender": "Gender",
    "Predicted Age": "Age",
    "Skin Tone": "Skin tone",
    "Intersection": "Intersection",
}
# Integer code -> readable subgroup name, per attribute
SUBGROUP_NAMES = {
    "Predicted Gender": {0: "Female", 1: "Male"},
    "Predicted Age": {
        0: "Child",
        1: "Youth",
        2: "Adult",
        3: "Middle-Aged",
        4: "Senior",
    },
    "Skin Tone": {i: f"Skin Tone {i}" for i in range(1, 11)},
    "Intersection": {
        0: "Female, Light",
        1: "Female, Medium",
        2: "Female, Dark",
        3: "Male, Light",
        4: "Male, Medium",
        5: "Male, Dark",
    },
}
# Same mapping with string codes ("1" -> "Male")
SUBGROUP_LABELS = {
    a: {str(k): v for k, v in m.items()} for a, m in SUBGROUP_NAMES.items()
}
# Name -> code, so subgroups sort in code order (Child, Youth, ... not alphabetical)
SUBGROUP_ORDER = {v: k for m in SUBGROUP_NAMES.values() for k, v in m.items()}
SUBGROUP_ORDER.update({"Annotated": 0, "Unannotated": 1})

# Pseudo-attribute for annotation status (buffer analysis, steps 1-4 only)
ANNOT_ATTR = "Annotation"
ALL_ATTRS = [*DEMOGRAPHICS, ANNOT_ATTR]
# All five annotation fields (metadata names); any value > 0 = annotated
ANNOTATION_FIELDS = [*DEMOGRAPHICS, "Skin Tone Group"]
# Memlog (snake_case) column -> attribute name
LOG_COLS = {
    "predicted_gender": "Predicted Gender",
    "predicted_age": "Predicted Age",
    "skin_tone": "Skin Tone",
    "intersection": "Intersection",
}
# All five annotation fields as named in the memlog CSVs
LOG_DEMO_FIELDS = [*LOG_COLS, "skin_tone_group"]

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
MODEL_LABELS = {"pina": "PINA", "soyo": "SOYO", "sprompts": "S-Prompts"}  # routing
METHOD_LABELS = {"derpp": "DER++", "gradmatch": "GradMatch"}  # replay buffer

# ---------------------------------------------------------------------------
# Column names and values in the input files
# ---------------------------------------------------------------------------
CKPT = "trained_through_task"
INSERT_EVENTS = {"insert", "replace"}  # events.csv events that open a residency
MISSING = {"", "nan", "none", "<na>"}  # cell values treated as missing
FAIR_COLS = ["FEO", "FDP", "FOAE", "FEOP"]
# mark_missing_demographics.py: memlog file -> uid columns, in order of preference
UID_COLS = {
    "events.csv": ["sample_uid", "image_path"],
    "sampling.csv": ["sample_uid"],
    "lifecycle.csv": ["uid"],
}
# Never overwritten by mark_missing_demographics.py, even if passed in --demo-cols
PROTECTED = {"target", "label", "image_path", "sample_uid", "uid", "source", "path"}

# ---------------------------------------------------------------------------
# Buffer tables
# ---------------------------------------------------------------------------
UNIT_METRICS = {  # buffer metric column -> label in the correlation tables
    "rep_ratio": "Representation",
    "exposure": "Exposure",
    "eviction_rate": "Eviction rate",
    "lifetime": "Lifetime",
}
DYN_METRICS = [  # dynamics table rows: (label, table key, value column, count column)
    ("Representation", "rep", "buffer_ratio", "n"),
    ("Retention", "ret", "retention", "n_prev"),
    ("Exposure", "task", "exposure", "at_risk"),
    ("Eviction", "task", "eviction_rate", "at_risk"),
]

# ---------------------------------------------------------------------------
# Buffer cache
# ---------------------------------------------------------------------------
# Bump when the per-run statistics change, so old caches are not reused.
CACHE_VERSION = 5
# Arguments that change the per-run results. Everything else (min-buffer, min-n,
# n-boot, digits, fairness files) is applied after the cache and never stale.
CACHED_ARGS = ("n_tasks", "draws", "n_null", "seed")
RUN_TABLES = (
    "rep",
    "rep_detail",
    "task",
    "ret",
    "rvr",
    "coh",
    "km",
    "units",
    *(f"{k}_{c}" for c in CLASSES for k in ("rep", "task", "ret")),
)

# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
ATTR_LABEL_OFFSET = 4  # extra gap (points) between subgroup names and attribute label
# Pastel colour maps (no red, no blue). Each sequential ramp keeps one hue and gets
# steadily darker (OKLCH L 0.97 -> 0.52/0.58); the diverging map runs from green
# through a near-colourless warm grey to violet, with equally deep ends.
PASTEL_RAMPS = {
    "violet": ["#f5f2fa", "#e2daf1", "#c9bbe6", "#ab97d4", "#8c74bf", "#6f58a3"],
    "green": ["#f2f8f3", "#d9eedf", "#b8dcc3", "#91c5a2", "#6aa982", "#4d8b65"],
}
PASTEL_DIVERGING = [
    "#5f9e76",
    "#9cc9aa",
    "#d3e8d9",
    "#efedea",
    "#e0d7f0",
    "#b3a2dc",
    "#8670c0",
]
NO_DATA = "#c9c5be"  # masked / empty cells: darker than any ramp start, no hue
# Colour bar: as tall as the heatmap; width = height / COLORBAR_ASPECT (larger = thinner)
COLORBAR_ASPECT = 45
COLORBAR_PAD = 0.015  # gap between heatmap and colour bar, fraction of the figure
# Trajectory line plots (plot_trajectories.py). Categorical hues in a fixed,
# colour-blind-checked order; the first three stay distinct even when every pair
# of lines overlaps, so larger groups add line style and marker as well.
LINE_PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
LINE_STYLES = ["-", "--", ":", "-."]
LINE_MARKERS = ["o", "s", "^", "D", "v", "P"]
# Colours per attribute: "reference" (as in the Fisher-similarity figures, see
# below), "categorical" (LINE_PALETTE / tone-group hues) or any matplotlib colour
# map name (plasma, viridis, cividis, magma, ...), sampled over TRAJ_CMAP_RANGE
# (plasma/viridis/magma end in pale yellow; 0.85 keeps it visible on white).
TRAJ_COLORS = {
    "Predicted Gender": "reference",
    "Predicted Age": "reference",
    "Skin Tone": "reference",
    "Intersection": "reference",
}
TRAJ_CMAP_RANGE = (0.0, 0.85)
# x-axis of the trajectory plots (--x): "cumulative" = subgroup's share of the
# training data of tasks 1..t (cumulative_share=True in the Fisher script),
# "dataset" = share of task t's data only, "overall", or "task"
TRAJ_X = "cumulative"
# Subgroup counts behind the x-axis of the Fisher panels (--fisher-x-counts):
# "fisher" = the n column of the Fisher results file, first n per subgroup and
# dataset (exactly as get_subgroup_n in the Fisher script); "metadata" =
# annotated training images. Memory and routing panels always use the metadata.
FISHER_X_COUNTS = "fisher"
# "reference" scheme
REF_GENDER_COLORS = {"Female": "#d62728", "Male": "#1f77b4"}
REF_AGE_CMAP, REF_AGE_RANGE = "plasma", (0.08, 0.9)  # Child (indigo) -> Senior (gold)
REF_AGE_STYLES = ["-", "--", ":", "-.", "-"]
# Skin tone 1 -> 10 runs along this ramp (light tan -> dark brown); gender x tone
# uses its Light / Medium / Dark points
# (brown_gradient of the Fisher script, sampled 0.15 -> 0.95 for tones 1-10 and
# at 0.25 / 0.55 / 0.85 for Light / Medium / Dark, as there)
REF_SKIN_RAMP = ["#f6e8c3", "#dfc27d", "#bf812d", "#8c510a", "#543005"]
REF_SKIN_RANGE = (0.15, 0.95)
REF_TONE_POINTS = (0.25, 0.55, 0.85)
REF_TONE_STYLES = {"Light": "-", "Medium": ":", "Dark": "--"}
# y-axis top for memory-share panels
MEMORY_YMAX = 0.6
# Vertical method-family label left of each grid row (consecutive rows of the
# same family share one label); keys are the row kinds of GRID_ROWS
ROW_FAMILIES = {
    "routing": "Prompt based",
    "routing_error": "Prompt based",
    "memory": "Memory based",
    "memory_annotated": "Memory based",
    "fisher": "Regularization based",
}
FAMILY_SIDE = "left"  # where the family labels go: "left" or "right" (--family-side)
# Grid figure rows, top to bottom: "routing:<model>", "routing_error:<model>",
# "memory:<method>", "memory_annotated:<method>", "fisher"
GRID_ROWS = ["memory_annotated:gradmatch", "routing:soyo", "fisher"]
# Fisher cosine-similarity results (fisher_evolution_pipeline.py via
# run_fisher.sh); first existing path is used
FISHER_CSV = [
    "results/consistency/empirical_fisher_local_distances_merged.csv",
    "src/curve/results/consistency/empirical_fisher_local_distances_merged.csv",
    "artifacts/results/consistency/empirical_fisher_local_distances_merged.csv",
]
# Skin-tone groups in order; hue per group for the skin tone and intersection panels
TONE_GROUPS = ["Light", "Medium", "Dark"]
# Heatmap style defaults (command-line flags override them)
STYLE_DEFAULTS = {
    "annotate": False,
    "fmt": "{:.2f}",
    "size": 5.0,
    "cmap": "violet",
    "hatch": "////",
    "hatch_color": "#8a8782",
    "hatch_alpha": 0.6,
    "hatch_width": 0.6,
}
RC_PARAMS = {
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "pdf.fonttype": 42,  # embed TrueType fonts (camera-ready friendly)
}

FISHER_X_COUNTS = "fisher"
FISHER_X_MODE = "cumulative"  # Fisher x-axis: "cumulative", "dataset" or "overall"
