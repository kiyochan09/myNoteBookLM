# OCR runtime layout

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

`engine_paths.py` is the only place that resolves the OCR runtime. Python
subprocesses run from the MyNotebookLM project root so the installed `ocr`
module is selected rather than a same-named source file from the standalone
application. PDFs, OCR output, dictionaries, and templates are stored under
the MyNotebookLM project's `data` directory.

The venv uses the Python 3.12 installation it was created from, as recorded in
its `pyvenv.cfg`; the OCR package and its models are stored in the local venv.
