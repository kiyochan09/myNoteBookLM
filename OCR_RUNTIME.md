# OCR runtime layout

Runtime discovery now uses `app/runtime_paths.py` and accepts both bundled and
sibling OCR environments. No user name or drive letter is fixed in code.
`MYNOTEBOOKLM_OCR_DIR`, `MYNOTEBOOKLM_OCR_PYTHON`, and
`MYNOTEBOOKLM_RESEARCH_PYTHON` support unusual placements; relative overrides use
the current application directory.

Copied interpreters are probed before use. Use `repair_runtime.py --ocr` or
`--research` with the same major/minor Python version to repair old base-Python
references without deleting packages or models. Research setup also checks and
repairs unusable launchers. New model manifests use relative paths.

See [別PCへの移行手順](別PCへの移行手順.md) for current supported layouts and
the full migration procedure. The sibling layout below remains supported.

MyNotebookLM and the standalone OCR Translator are kept under the same
Antigravity `scratch` directory. MyNotebookLM locates the shared OCR runtime
relative to its own source tree; it does not rely on a clone under
`C:\Users\...\source\repos` or on PATH lookup.

```text
antigravity/
└── scratch/
    ├── knowledge_base_system/
    │   ├── app/ocr_pipeline/engine_paths.py  # single path definition
    │   └── data/
    │       ├── user_dictionary.json
    │       ├── tcy_registry.json             # created when first saved
    │       └── tcy_templates/
    └── OCR-REPOS/ocr_engine/
        └── venv/                              # local NDLOCR package/runtime
```

`engine_paths.py` selects the working OCR runtime through `runtime_paths.py`. Python
subprocesses run from the MyNotebookLM project root so the installed `ocr`
module is selected rather than a same-named source file from the standalone
application. PDFs, OCR output, dictionaries, and templates are stored under
the MyNotebookLM project's `data` directory.

The venv uses the Python 3.12 installation it was created from, as recorded in
its `pyvenv.cfg`; the OCR package and its models are stored in the local venv.
