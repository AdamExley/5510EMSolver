# ECE 5510 2D FEM Waveguide Simulator

[GitHub Repository](https://github.com/AdamExley/5510EMSolver)

## Using the simulator

### Install dependencies

Requires [Python 3](https://www.python.org/downloads/)

```bash
pip install -r requirements.txt
```

### Run the default simulator
```bash
python sim.py
```
Automatically runs the `uStrip.in` geometry at 1 GHz and plots the electric field distribution.

Does simple mesh resampling to get a finer mesh for the simulation.

### Run simulator with options
```bash
usage: sim.py [-h] [--mode {profile,modes}] [--input {parallelPlateWG.in,rectWG.in,stripLine.in,uStrip.in}] [--freq FREQ] [--resamples RESAMPLES] [--grid GRID] [--n_eigen N_EIGEN]

optional arguments:
  -h, --help            show this help message and exit
  --mode {profile,modes}, -m {profile,modes}
                        Mode to run in. profile: Compute propagation and mode profile at a specific frequency modes: (NOT FULLY WORKING) Compute the propagation constant of the first N_EIGEN modes up to FREQ
  --input {parallelPlateWG.in,rectWG.in,stripLine.in,uStrip.in}, -i {parallelPlateWG.in,rectWG.in,stripLine.in,uStrip.in}
                        Input file
  --freq FREQ, -f FREQ  Frequency to solve at (Default 1e9)
  --resamples RESAMPLES, -r RESAMPLES
                        Number of resamples (Default 3)
  --grid GRID, -g GRID  Field display grid size (Default 50)
  --n_eigen N_EIGEN, -n N_EIGEN
                        Number of eigenvalues to compute (Default 2)
```


## Results

### Basic Problem

The base problem was that given in the project description.

1cm uStrip with 1cm ground plane spacing and 1cm padding on each side. epsilon_r = 2 for the substrate. 1GHz frequency.



![uStrip Simulation Result](results/base.svg)

The propagation constant of this result is higher than expected, but the mode profile is as expected.

Approximations give `eps_eff` around 1.6 although this simulation gives `eps_eff` close to 2.

Further testing showed that this is likely due to insufficient padding around the uStrip, as the field magnitude is still high at the edges of the simulation domain. A larger domain would likely give a more accurate result.

#### Upscaled

The same simulation was run with 5x resampling to give a finer mesh.

![uStrip Simulation Result](results/base_upscale.svg)