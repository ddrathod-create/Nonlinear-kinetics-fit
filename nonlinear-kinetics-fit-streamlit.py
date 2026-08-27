import io
from decimal import Decimal

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "sans-serif"

# ---------------- Dark theme (Catppuccin-ish, matching the original QSS) ----------------
DARK_CSS = """
<style>
.stApp {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}
div.stButton > button {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #89b4fa;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 15px;
}
div.stButton > button:hover {
    background-color: #45475a;
    color: #cdd6f4;
    border-color: #89b4fa;
}
div.stDownloadButton > button {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #89b4fa;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 15px;
}
div.stDownloadButton > button:hover {
    background-color: #45475a;
}
.results-card {
    background-color: #282a3a;
    border: 1px solid #89b4fa;
    border-radius: 10px;
    padding: 14px 20px;
}
.results-title {
    color: #89b4fa;
    font-size: 16px;
    font-weight: 600;
    padding-bottom: 6px;
}
.results-content {
    color: #cdd6f4;
    font-size: 16px;
    line-height: 170%;
}
.status-hint {
    color: #a6e3a1;
    font-size: 12px;
}
label, .stSlider label, .stMarkdown, p {
    color: #cdd6f4 !important;
}
</style>
"""

# Default initial guesses used for the very first fit
DEFAULT_P0 = {"dA": 0.1, "k": 0.02, "c": 0.168}


def exp_decay(t, dA, k, c):
    return dA * np.exp(-k * t) + c


def format_plain(value, sig_figs=4):
    """Format a number in plain decimal notation (never scientific), with sig_figs significant digits."""
    if value == 0:
        return "0"
    d = Decimal(str(float(value)))
    exponent = d.adjusted()  # position of the most significant digit
    decimal_places = max(sig_figs - exponent - 1, 0)
    formatted = f"{float(value):.{decimal_places}f}"
    return formatted


def load_csv(file_obj):
    """Load a tab-delimited (t, A) CSV, trying a few common encodings."""
    data = None
    for encoding in ("utf-8", "utf-16", "latin1"):
        try:
            file_obj.seek(0)
            data = pd.read_csv(file_obj, sep="\t", encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if data is None:
        raise ValueError("Could not read file with utf-8, utf-16, or latin1 encoding.")

    # Drop any extra trailing column (e.g. from a stray tab at line end)
    data = data.iloc[:, :2]
    data.columns = ["t", "A"]
    data["t"] = pd.to_numeric(data["t"], errors="coerce")
    data["A"] = pd.to_numeric(data["A"], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    return data


def run_fit(data, n1, n2, last_popt):
    """Fit exp_decay over points n1..n2 (1-indexed, inclusive). Returns a result dict."""
    subset = data.iloc[n1 - 1 : n2]
    t_fit = subset["t"].to_numpy()
    A_fit = subset["A"].to_numpy()

    if last_popt is not None:
        p0 = last_popt
    else:
        p0 = [DEFAULT_P0["dA"], DEFAULT_P0["k"], DEFAULT_P0["c"]]

    popt, pcov = curve_fit(exp_decay, t_fit, A_fit, p0=p0, maxfev=10000)

    dA_fit, k_fit, c_fit = popt
    perr = 2 * np.sqrt(np.diag(pcov))
    dA_err, k_err, c_err = perr
    tau = 1 / k_fit

    residuals = A_fit - exp_decay(t_fit, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((A_fit - np.mean(A_fit)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "popt": popt,
        "t_fit": t_fit,
        "A_fit": A_fit,
        "dA_fit": dA_fit, "k_fit": k_fit, "c_fit": c_fit,
        "dA_err": dA_err, "k_err": k_err, "c_err": c_err,
        "tau": tau,
        "r_squared": r_squared,
    }


def make_plain_text(result, n1, n2):
    return (
        f"\u0394A  = {format_plain(result['dA_fit'])} \u00b1 {format_plain(result['dA_err'], 2)}\n"
        f"k   = {format_plain(result['k_fit'])} \u00b1 {format_plain(result['k_err'], 2)} s\u207b\u00b9\n"
        f"A_e = {format_plain(result['c_fit'])} \u00b1 {format_plain(result['c_err'], 2)}\n"
        f"\u03c4   = {format_plain(result['tau'])} s\n"
        f"R\u00b2  = {result['r_squared']:.5f}\n"
        f"(fit range: points {n1} to {n2})"
    )


def make_html_results(result):
    return (
        f"<div class='results-card'>"
        f"<div class='results-title'>Fit Results</div>"
        f"<div class='results-content'>"
        f"&Delta;A = {format_plain(result['dA_fit'])} &plusmn; {format_plain(result['dA_err'], 2)}<br>"
        f"k = {format_plain(result['k_fit'])} &plusmn; {format_plain(result['k_err'], 2)} s<sup>-1</sup><br>"
        f"A<sub>e</sub> = {format_plain(result['c_fit'])} &plusmn; {format_plain(result['c_err'], 2)}<br>"
        f"&tau; = {format_plain(result['tau'])} s"
        f"</div></div>"
    )


def build_figure(data, t_fit, popt):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        data["t"], data["A"], "o",
        color="#2e7d32", markersize=4, markeredgewidth=0, alpha=0.85, label="Data",
    )

    t_smooth = np.linspace(t_fit.min(), t_fit.max(), 300)
    ax.plot(
        t_smooth, exp_decay(t_smooth, *popt),
        color="#d81b60", linewidth=2.2, label="Fit",
    )

    ax.axvspan(t_fit.min(), t_fit.max(), color="#89b4fa", alpha=0.12, zorder=0)

    ax.set_xlabel("Time (s)", fontsize=12, color="#1e1e2e")
    ax.set_ylabel("Absorbance", fontsize=12, color="#1e1e2e")
    ax.tick_params(direction="in", top=True, right=True, colors="#1e1e2e")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, loc="upper right")
    for spine in ax.spines.values():
        spine.set_color("#1e1e2e")

    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Nonlinear Kinetics Fit", layout="wide")
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.session_state.data = None
    if "last_popt" not in st.session_state:
        st.session_state.last_popt = None
    if "loaded_filename" not in st.session_state:
        st.session_state.loaded_filename = None

    header_col1, header_col2, header_col3 = st.columns([2, 1, 2])
    with header_col1:
        st.markdown("### Nonlinear Kinetics Fit")
    with header_col2:
        uploaded_file = st.file_uploader("Open CSV", type=["csv"], label_visibility="collapsed")
    with header_col3:
        fname = st.session_state.loaded_filename or "No file loaded"
        st.markdown(f"<span style='font-size:14px'>{fname}</span>", unsafe_allow_html=True)

    if uploaded_file is not None and uploaded_file.name != st.session_state.loaded_filename:
        try:
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            st.session_state.data = load_csv(file_bytes)
            st.session_state.loaded_filename = uploaded_file.name
            st.session_state.last_popt = None  # reset guess history for new dataset
            # Reset slider positions for the new dataset
            n_points = len(st.session_state.data)
            st.session_state.n1 = 1
            st.session_state.n2 = n_points
        except Exception as e:
            st.error(f"Failed to load file:\n\n{e}")
            st.session_state.data = None

    if st.session_state.data is None:
        st.info("Open a CSV file to begin.")
        return

    data = st.session_state.data
    n_points = len(data)

    # ---------------- Layout: plot on the left, controls + results on the right ----------------
    plot_col, side_col = st.columns([2.2, 1], gap="large")

    with side_col:
        st.markdown("**Fit range**")
        n1 = st.slider("Start point", min_value=1, max_value=n_points - 3,
                        value=st.session_state.get("n1", 1), key="n1_slider")
        n2 = st.slider("End point", min_value=4, max_value=n_points,
                        value=st.session_state.get("n2", n_points), key="n2_slider")

        # Enforce minimum gap of 3, mirroring the Qt on_slider_change logic
        if n2 - n1 < 3:
            if n1 != st.session_state.get("n1", 1):
                n2 = min(n1 + 3, n_points)
            else:
                n1 = max(n2 - 3, 1)

        st.session_state.n1 = n1
        st.session_state.n2 = n2

        try:
            result = run_fit(data, n1, n2, st.session_state.last_popt)
        except RuntimeError as e:
            st.error(f"Fit failed for this range:\n\n{e}")
            return

        st.session_state.last_popt = result["popt"]

        st.markdown(make_html_results(result), unsafe_allow_html=True)
        st.markdown(
            f"<p class='status-hint'>Fit converged &nbsp; R\u00b2 = {result['r_squared']:.5f}</p>",
            unsafe_allow_html=True,
        )

        # ---------------- Export buttons ----------------
        plain_text = make_plain_text(result, n1, n2)
        fig = build_figure(data, result["t_fit"], result["popt"])

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")

        b1, b2 = st.columns(2)
        with b1:
            st.download_button(
                "Export Results",
                data=plain_text.encode("utf-8"),
                file_name="fit_results.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with b2:
            st.download_button(
                "Save Plot (PNG)",
                data=png_buf.getvalue(),
                file_name="fit_plot.png",
                mime="image/png",
                use_container_width=True,
            )

    with plot_col:
        st.pyplot(fig, use_container_width=True)

    plt.close(fig)


if __name__ == "__main__":
    main()
