# Whistling as a Proportional Switch: data and analysis scripts

Data and analysis scripts accompanying the paper *Whistling as a Proportional
Switch: Design, Instrumentation, and Empirical Comparison of Pitch- and
Intensity-Based Continuous Control* (IEEE Access).

Twenty-four participants controlled a virtual force-sensitive switch by
whistling, once with the pitch and once with the intensity (tone amplitude) of
the whistle. With each channel they held static target forces (maintenance
task) and followed a sinusoidal target (tracking task). The package holds every
scored trial, the complete per-tick record of each trial, the calibration,
open-loop and session logs, the questionnaire responses, and the scripts that
compute the results of the paper. Participants appear only as codes.

## Contents

| Path | Contents |
|---|---|
| `participant tests/` | Main study (24 participants): per-trial and per-tick files for each channel and task, calibration, open-loop and session logs, and the questionnaire workbook |
| `pilot tests/` | The same files for the three pilot participants |
| `scripts/` | Python scripts that compute the results of the paper; see [scripts/README.md](scripts/README.md) |
| [COLUMNS.md](COLUMNS.md) | File layout and the meaning of every column |
| [questionnaires_tr.md](questionnaires_tr.md) | Turkish wording of the questionnaires as administered |
| [LICENSE.md](LICENSE.md) | Licenses: data CC BY 4.0, scripts MIT |

## Reproducing the results

Install Python 3 with NumPy, SciPy, pandas, statsmodels, Matplotlib and
openpyxl, then run a script from the `scripts/` folder, for example:

```
cd scripts
python maintenance_tracking.py
```

Each script prints its results. [scripts/README.md](scripts/README.md) lists
the sections, tables and figures of the paper that each script reproduces.

## License

The data are licensed under CC BY 4.0 and the scripts under the MIT License;
see [LICENSE.md](LICENSE.md).
