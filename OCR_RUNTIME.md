# OCR runtime layout

MyNotebookLM owns one OCR installation under its application directory:

```text
myNoteBookLM/
├── app/ocr_pipeline/          # layout, tables, image processing, integration
├── runtimes/ocr/
│   ├── src/                  # managed, locally modified OCR implementation
│   │   ├── ocr.py
│   │   ├── deim.py / parseq.py / ndl_parser.py
│   │   ├── reading_order/ / tcy_wrapper.py
│   │   ├── config/
│   │   └── model/            # detector and three recognition models
│   ├── venv/                 # Python 3.12 and preserved dependency packages
│   ├── dependencies.lock.txt
│   ├── model-manifest.json
│   ├── repair_launchers.py
│   └── licenses/
└── data/                     # PDFs, OCR output, dictionaries, user data
```

`app/runtime_paths.py` defines this one location. `engine_paths.py` runs
`runtimes/ocr/venv/Scripts/python.exe -m ocr` from the application root.
The relative `mynotebooklm_ocr.pth` file selects `runtimes/ocr/src`.
The custom source and models are not hidden inside dependency site-packages.

The sibling OCR-REPOS application, its source tree, and old hard-coded resource
paths are not searched. MYNOTEBOOKLM_OCR_DIR and MYNOTEBOOKLM_OCR_PYTHON are no
longer used to select an external installation. The separate research runtime
and MYNOTEBOOKLM_RESEARCH_PYTHON retain their existing behavior.

Use the matching Python 3.12 to run `repair_runtime.py --ocr` after moving the
application to a new PC or folder. This rebuilds interpreter configuration,
activation scripts, and installed command launchers without deleting dependency
packages, OCR source, or models. A venv folder alone is not a portable executable:
see the [Python venv documentation](https://docs.python.org/3.12/library/venv.html).
`check_environment.py` verifies the bundled runtime and the four model files.

The original NDLOCR-Lite license notices are retained in `runtimes/ocr/licenses`.
The source includes the app-specific corrections already in use before
consolidation. Do not replace it by reinstalling a different ndlocr-lite package.
See [別PCへの移行手順](別PCへの移行手順.md) for the complete application migration.