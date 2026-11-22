# Nonliner Kinetics Fit 
ΔT Analysis (Bomb Calorimetry)

An app for the CHM 343 physical chemistry lab. It determines ΔT from a bomb calorimetry temperature vs. time run by extrapolating the initial and final temperature drifts to a chosen time. 

**Use it online:**  https://bomb-calorimetry-deltat-klkamv5ydjd9ztcjyguojj.streamlit.app/

## Data file

Two columns: time (s) and temperature

## How to use

1. Load your data file.
2. Adjust the four sliders (all in seconds):
   - **tmin** is where the initial-period fit starts.
   - **ti** is where the initial-period fit ends.
   - **td** is the time at which ΔT is read.
   - **tf** is where the final-period fit starts (it runs to the end of the data).
3. The app fits a straight line to the initial period (tmin → ti) and another to the final period (tf → end). ΔT is the difference between the two lines at `td`.
4. Read the results in the **ΔT Results** box: ΔT, Tᵢ, T_f, both slopes, Area₁ and Area₂.
5. Export the results as a text file (it also records your slider values) and save the plot as an image.

## What the plot shows

- The data, with both fitted lines extended across the run.
- Dotted vertical lines at `ti` and `tf`.
- A red bracket at `td` showing ΔT.
- Area 1 (blue) is the area between the data and Tᵢ from ti to td. Area 2 (orange) is the area between T_f and the data from td to tf. Choosing `td` so the two areas are equal is the usual way to pick it.

## Notes

- The sliders start at tmin = 80 s, ti = tmin + 280 s, td = ti + 80 s and tf = (end of run) − 400 s. Adjust them to suit your own run.
- Temperatures and ΔT are labeled in °C. If your data is in another unit, change the labels in the code.
- Each fit period needs at least 2 data points. If the sliders are too close together, the app tells you.
- Results are shown in plain decimals, with no scientific notation.

## Running the desktop version (optional)

You need Python 3 and these packages:

```
pip install PyQt5 numpy pandas matplotlib
```

Then, in a terminal in the folder containing the file:

```
python delta_t_analysis_app.py
```

If you see `ModuleNotFoundError: No module named 'PyQt5'`, run the `pip install` command above.



Thank you, Professor Wolf, for sharing the Mathematica notebook for bomb calorimetry ΔT analysis. This Python version was built from it!
