# Reproduce the bAbI pilot

Use a fresh BABI_PILOT_OUTPUT directory. Python must provide PyTorch. Download the archive URL recorded in protocol.json, place it under raw/tasks.tar.gz, and extract the en-10k qa1, qa2 and qa4 train/test text files into raw/. Preserve the source file names. Extraction must not place arbitrary archive paths outside the output directory.

The stages below reflect the actual exploratory chronology, including unsuccessful variants. For review, the complete results and per-epoch histories are already saved. Do not rerun training against this study's now-observed test results.

```sh
python3 scripts/pilot-babi.py prepare
python3 scripts/pilot-babi.py train
python3 scripts/prepare-babi-bridge.py
python3 scripts/pilot-babi.py train --tasks 0,1,2 --models counts-mlp,mean,ordered-mlp,sentence-mlp,memory-1,memory-2,memory-3
python3 scripts/prepare-babi-twofact-bridge.py
python3 scripts/pilot-babi.py train --tasks 100 --models ordered-mlp,sentence-mlp,memory-1,memory-2,memory-3,transformer-2
python3 scripts/pilot-babi.py train --tasks 1,100,2 --models question-transformer-1,question-transformer-2
python3 scripts/pilot-babi.py train --tasks 1,100,2 --models retrieval-1,retrieval-2,retrieval-3
python3 scripts/pilot-babi.py train --tasks 100 --models retrieval-2,retrieval-3 --epochs 100 --patience 100 --tag extended
python3 scripts/pilot-babi.py train --tasks 100 --models memory-2 --support-supervision --tag assisted
python3 scripts/pilot-babi.py train --tasks 0 --models counts-mlp,mean,ordered-mlp --seeds 211,307
python3 scripts/pilot-babi.py train --tasks 1 --models ordered-mlp,sentence-mlp,memory-1,counts-mlp,mean --seeds 211,307
python3 scripts/pilot-babi.py train --tasks 4 --models counts-mlp,ordered-mlp,transformer-1,transformer-2 --seeds 211,307
python3 scripts/pilot-babi.py train --tasks 100 --models memory-1,memory-2 --seeds 211,307
python3 scripts/pilot-babi.py train --tasks 100 --models memory-2 --seeds 211,307 --support-supervision --tag assisted
python3 scripts/pilot-babi.py train --tasks 100 --models retrieval-2,retrieval-3 --seeds 211,307 --epochs 100 --patience 100 --tag extended
python3 scripts/pilot-babi.py train --tasks 100 --models retrieval-1 --seeds 137,211,307 --epochs 100 --patience 100 --tag extended
python3 scripts/pilot-babi.py train --tasks 2 --models memory-1,memory-2 --seeds 211,307
python3 scripts/test_babi_pilot.py
python3 scripts/freeze-evaluate-babi.py
python3 scripts/report-babi-pilot.py
```

Standard settings: 30 epochs maximum, patience 5. Extended is a distinct 100-epoch budget with patience 100. Auxiliary fact supervision is used only in runs tagged assisted and only on the synthetic two-fact bridge. Raw test files never enter training. Results are averaged across seeds, not ensembled.

App delivery is intentionally a subsequent step: these prototypes expose the numerical operations but do not implement multiclass QA import or Gradescope submission/grading in the browser.
