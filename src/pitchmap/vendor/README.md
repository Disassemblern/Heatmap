# Vendored code

Third-party sources copied into the repo so the pipeline does not depend on
unpackaged research repositories. Files here are exempt from the project's file
size and style conventions: they stay as close to upstream as possible so they
can be diffed against it. Every local change is listed in the file's header.

| Path | Upstream | Commit / version | Licence |
| --- | --- | --- | --- |
| `osnet.py` | KaiyangZhou/deep-person-reid, `torchreid/models/osnet.py` | torchreid 0.2.5 | MIT |
| `deep_eiou/` | hsiangwei0903/Deep-EIoU, `Deep-EIoU/tracker/` | `2fc7d270` | none published |
| `gta_link/` | sjc042/gta-link | `e4d5cc40` | MIT |

`deep_eiou/` has no upstream licence. It is vendored here for personal,
non-commercial experimentation only. Do not redistribute that directory. Paper:
Huang et al., WACV 2024 Workshops (Iterative Scale-Up ExpansionIoU).

`osnet.py` exists because importing the `torchreid` package pulls its training
stack (tensorboard, gdown, ...) at import time; the architecture file itself
imports only torch.

Local modification policy: import-path fixes, numpy 2 compatibility
(`np.float` to `float`), and delete-only removal of CLI or visualisation code.
No behavioural edits.
