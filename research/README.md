# research/

Notes that cut across models and kits: techniques, prior art and measurements we collect before a kit needs them. A kit's own model
knowledge stays in `kits/<model>/NOTES.md`; what several kits share lives here. Every number names its source; numbers from other
machines are tagged as theirs.

| note | topic | kits it serves |
|---|---|---|
| [nvme-streaming.md](nvme-streaming.md) | weights and tables larger than memory: n-gram (Engram / PLE) tables and MoE experts read from NVMe on a unified-memory APU | hy4-flash, Qwen3.8 Flash Next, any MoE quant beyond RAM |
