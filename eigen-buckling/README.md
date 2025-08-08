
# Finite Element Modeling Framework for Stiffened Panels (Hydrostatic Load Case)

This project provides a modular Python framework to evaluate stiffened panel structures under hydrostatic loading using ABAQUS. It separates input/output data structures, ABAQUS modeling logic, and workflow orchestration into distinct, reusable components.

---

## 📁 Project Structure

```
.
├── run_finite_element_model.py         # Example script for executing an analysis
├── models/
│   ├── hydrostatic.py                  # ABAQUS model for stiffened panel under hydrostatic pressure
│   └── IO_hydrostatic.py              # Input/output dataclasses for the hydrostatic model
├── utils/
│   └── FiniteElementModel.py          # Generic finite element model interface
├── data/
│   └── hydrostatic/
│       ├── input.jsonl                # Input records (JSON Lines format)
│       └── output.jsonl               # Output records (JSON Lines format)
```
---

## ⚙️ How It Works

1. **Define Input Geometry & Properties**  
   Using the `PanelInput` class from `IO_hydrostatic.py`, define the global panel geometry, stiffener dimensions, material thicknesses, and mesh controls.

2. **Initialize Model Runner**  
   The `FiniteElementModel` class handles writing input data, calling the ABAQUS model, and reading the output.

3. **Run the Model**  
   `run_finite_element_model.py` demonstrates how to run a simulation, including:
   - Writing structured input to a `.jsonl` file
   - Launching ABAQUS using `subprocess`
   - Parsing results from the `.odb` file and writing output as JSON

---

## 📥 Input Format

Defined in `PanelInput`:
```python
@dataclass
class PanelInput:
    id: str
    num_transverse: int
    num_longitudinal: int
    width: float
    length: float
    t_panel: float
    t_transverse_web: float
    t_transverse_flange: float
    t_longitudinal_web: float
    t_longitudinal_flange: float
    h_transverse_web: float
    h_longitudinal_web: float
    w_transverse_flange: float
    w_longitudinal_flange: float
    pressure_magnitude: float
    mesh_plate: float
    mesh_transverse_web: float
    mesh_transverse_flange: float
    mesh_longitudinal_web: float
    mesh_longitudinal_flange: float
```

---

## 📤 Output Format

Defined in `PanelOutput`:
```python
@dataclass
class PanelOutput:
    id: str
    max_stress: float
    assembly_mass: float
    element_counts: Dict[str, int]
    stress_field: List[ElementStress]
    job_name: str
    step: str
```

---

## 🚀 Running the Model

To run a simulation with your custom input:
```bash
abaqus cae noGUI=models/hydrostatic.py
```

Or from Python:
```bash
python run_finite_element_model.py
```

This script constructs a `PanelInput` object, runs the model, and logs the results to the JSON Lines files.

---

## 🧩 Extensibility

- The `FiniteElementModel` class is generic and can support other models by swapping out `PanelInput`, `PanelOutput`, and the ABAQUS `.py` script.
- The I/O structure is fully JSON-serializable, enabling integration with surrogate modeling pipelines or optimization frameworks.

---

## 📄 File I/O

- **Inputs**: Appended to `data/hydrostatic/input.jsonl`
- **Outputs**: Retrieved from `data/hydrostatic/output.jsonl`
- Supports `.gz` compression if required for large-scale usage

---

## 🛠 Requirements

- **ABAQUS** (running in CAE noGUI mode)
- Python 3.8+ (for general code)
- Python 2.7 (for ABAQUS scripting environment)
- `numpy`, `json`, `gzip`

---

## 🔍 Example Usage

```python
from utils.FiniteElementModel import FiniteElementModel
from models.IO_hydrostatic import PanelInput, PanelOutput

panel_input = PanelInput(
    id="",
    num_transverse=3,
    num_longitudinal=6,
    width=6.0,
    length=9.0,
    t_panel=0.025,
    t_transverse_web=0.025,
    t_transverse_flange=0.025,
    t_longitudinal_web=0.025,
    t_longitudinal_flange=0.025,
    h_transverse_web=0.45,
    h_longitudinal_web=0.40,
    w_transverse_flange=0.25,
    w_longitudinal_flange=0.175,
    pressure_magnitude=-10000,
    mesh_plate=0.05,
    mesh_transverse_web=0.05,
    mesh_transverse_flange=0.05,
    mesh_longitudinal_web=0.05,
    mesh_longitudinal_flange=0.05
)

fem = FiniteElementModel("models/hydrostatic.py", "data/hydrostatic/input.jsonl", "data/hydrostatic/output.jsonl", PanelInput, PanelOutput)
result = fem.evaluate(panel_input)
```

## 🔍 Example Usage - Continued Evaluations

```python

# -------------------------------
# Static / baseline input values
# -------------------------------
base_input = PanelInput(
    id="",  # will be overwritten
    num_transverse=3,
    num_longitudinal=6,
    width=6.0,
    length=9.0,
    t_panel=0.025,
    t_transverse_web=0.025,
    t_transverse_flange=0.025,
    t_longitudinal_web=0.025,
    t_longitudinal_flange=0.025,
    h_transverse_web=0.45,
    h_longitudinal_web=0.40,
    w_transverse_flange=0.25,
    w_longitudinal_flange=0.175,
    pressure_magnitude=-10000,
    mesh_plate=0.05,
    mesh_transverse_web=0.05,
    mesh_transverse_flange=0.05,
    mesh_longitudinal_web=0.05,
    mesh_longitudinal_flange=0.05,
)

for i, sample in enumerate(tqdm(samples, desc="Evaluating panel samples")):
    new_input = replace(
        base_input,
        id=str(uuid4()),
        num_transverse=int(sample[0]),
        num_longitudinal=int(sample[1]),
        t_panel=sample[2],
        t_transverse_web=sample[3],
        t_transverse_flange=sample[4],
        t_longitudinal_web=sample[5],
        t_longitudinal_flange=sample[6],
    )

    try:
        result = fem.evaluate(new_input)
    except Exception as e:
        print(f"[{i}] Evaluation failed: {e}")
```

---

## 🧪 Troubleshooting

- Make sure `setPath` in `hydrostatic.py` matches your current working directory.
- Ensure that ABAQUS is callable from the command line (`abaqus cae noGUI=...`).
- If no output is written, check `abaqus_log.txt` for errors.
- Be aware that node/face indexing in ABAQUS may change if model geometry is altered.

---

## 🧱 Development Tips

- Use `writeDebug()` in `hydrostatic.py` to output intermediate values for quick inspection.
- Modify `PanelOutput.to_dict()` if new results are added to the ABAQUS model.
- Structure all input/output in JSON Lines format for easy integration with ML or optimization workflows.

---

## 🤖 Integration with Surrogate Modeling

- Outputs are JSON-serializable and easily parsed using pandas or other ML tools.
- The `FiniteElementModel.save_history()` method can be used to log batch evaluations for dataset creation.

---

