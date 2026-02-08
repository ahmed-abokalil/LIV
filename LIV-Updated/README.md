# Drell-Yan Lorentz Violation Analysis (M4 Optimized)

##  Technical Stack & Environment

This is an updated, faster, and more precise version specifically optimized for Apple Silicon. It utilizes the latest library releases compatible with **Python 3.12** in a Conda environment.

### Core Dependencies
| Library | Version | Purpose |
| :--- | :--- | :--- |
| **NumPy** | `1.26+` | Vectorized array operations |
| **SciPy** | `1.14+` | Numerical integration using `quad` and `simpson` |
| **PyTorch** | `2.x` | High-speed tensor contractions via `tn.einsum` |
| **LHAPDF** | `6.5+` | Parton Distribution Function (PDF) interpolation |
| **Matplotlib**| `3.8+` | Scientific visualization of LIV contributions |

---

## M4 Optimizations

### 1. Multiprocessing Architecture
On macOS (Sonoma+), the `multiprocessing` library uses the **`spawn`** method. This repository implements a decoupled `functions.py` module to allow worker processes to inherit the physics namespace correctly.

### 2. Numerical Convergence Tuning
The simulation handles the sharp **Z-boson resonance peak** near $Q \approx 91$ GeV by dynamically scaling integration limits.
* **Integration Depth**: Default `limit` in `scipy.integrate.quad` is increased to **100-200** to eliminate "maximum subdivisions achieved" errors.


---

## Installation

```bash
# Create the optimized environment
conda create -n liv python=3.12
conda activate liv

# Install dependencies
conda install numpy scipy matplotlib pytorch
# Ensure LHAPDF is installed and PDF sets (NNPDF31) are downloaded
