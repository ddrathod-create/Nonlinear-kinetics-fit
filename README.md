# Nonlinear Kinetics Fit

An app for the CHM 343 physical chemistry lab. It fits an exponential decay to absorbance vs. time data and reports the rate constant `k` and the time constant `τ`. 
**Use it online:** https://nonlinear-kinetics-fit-djg7c9ibgjquszv42dn29s.streamlit.app/

## The model

```
A(t) = ΔA · exp(−k·t) + Aₑ
```

- `ΔA` is the total change in absorbance.
- `k` is the rate constant (s⁻¹).
- `Aₑ` is the absorbance the curve levels off at.
- `τ = 1/k` is the time constant (s).

## Data file

Two columns: time (s) and absorbance. The file should be tab-delimited with a header row. 

## How to use

1. Load your data file.
2. Use the **n1** and **n2** sliders to choose which data points (by index) are included in the fit. The shaded band on the plot shows the selected range. The fit and results update live as you move them.
3. Read the results in the **Fit Results** box: ΔA, k, Aₑ and τ, each with its uncertainty (±2 standard errors, roughly a 95% interval). R² for the selected range is also shown.
4. Export the results as a text file and save the plot as an image.

## Notes

- The first fit starts from the guesses ΔA = 0.1, k = 0.02 and Aₑ = 0.168. After that, each new fit starts from the previous result, as in the notebook.
- At least 4 data points must be selected so the fit has enough data.
- If a fit fails for some range, the app says so. Try a different range.
- Results are shown in plain decimals, with no scientific notation.

## Running the desktop version (optional)

You need Python 3 and these packages:

```
pip install PyQt5 numpy pandas scipy matplotlib
```

Then, in a terminal in the folder containing the file:

```
python kinetics_fit_app.py
```

If you see `ModuleNotFoundError: No module named 'PyQt5'`, run the `pip install` command above.

Thank you, Professor Wolf, for sharing the Mathematica notebook. This Python version was built from it!
