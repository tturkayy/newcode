# figures & diagrams

this directory contains architecture diagrams and technical schematics for the **newcode** protocol.

### files

* `diagram_overview.png` / `_sepia.png` — end-to-end transmitter, optical channel, and receiver pipeline.
* `diagram_tx.png` / `_sepia.png` — transmitter stage from utf-8 payload to 2d wave stream.
* `diagram_modulation.png` / `_sepia.png` — 16-apsk (4x4 pam-4/qpsk) constellation and continuous trace.
* `diagram_rx.png` / `_sepia.png` — receiver stage with centroid tracking and 1d cnn inference.

all assets are generated programmatically via `generate_all_diagrams.py` to match the web colorways.
