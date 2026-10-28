import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- EDIT THESE PATHS ----------

EXPERIMENT_FILE = Path("/Users/chinaecheremanyanwu/Downloads/polydisperseCapillaryBreakup-main/Data/ThinningCurves_PS7_PS16_andMixtures.xlsx")

# Add one trajectory file for each completed concentration.
# Leave missing concentrations out.
BD_PS16 = {
    5:  "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=3.45_v=0_p=100_2R0=1.24mm_MW=14.96M_c_cstar=0.001931/polymer_100.xyz",
    10: "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=3.45_v=0_p=100_2R0=1.24mm_MW=14.96M_c_cstar=0.003861/polymer_100.xyz",
    20: "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=3.45_v=0_p=100_2R0=1.24mm_MW=14.96M_c_cstar=0.007722/polymer_100.xyz",
    50: "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=3.45_v=0_p=100_2R0=1.24mm_MW=14.96M_c_cstar=0.01931/polymer_100.xyz",
    100: "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=3.45_v=0_p=100_2R0=1.24mm_MW=14.96M_c_cstar=0.03861/polymer_100.xyz",
    200: "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=3.45_v=0_p=100_2R0=1.24mm_MW=14.96M_c_cstar=0.07722/polymer_100.xyz",
}

BD_PS7 = {
    200:  "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=2.41_v=0_p=100_2R0=1.24mm_MW=6.848M_c_cstar=0.05102/polymer_100.xyz",
    500:  "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=2.41_v=0_p=100_2R0=1.24mm_MW=6.848M_c_cstar=0.1276/polymer_100.xyz",
    1000: "/Users/chinaecheremanyanwu/Desktop/MyBD_Results/P=100/N=11/CaBER_visEq/MPI_Hookean_FD=2.41_v=0_p=100_2R0=1.24mm_MW=6.848M_c_cstar=0.2551/polymer_100.xyz",
}

# Optional manually inspected crossover times, in seconds.
# Otherwise, use the first local maximum of saved strain rate,
# following the MATLAB definition.
TC_OVERRIDE = {
    # ("PS16", 50): 0.047,
}

# ---------- READ BD OUTPUT ----------

def read_bd(filepath):
    filepath = Path(str(filepath).strip().strip("'\"")).expanduser()
    rows = []

    with filepath.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue

            beads = int(line)
            header = f.readline()
            values = dict(
                token.split("=", 1)
                for token in header.split()
                if "=" in token
            )

            rows.append([
                float(values["tVEC"]),
                float(values["d"]),
                float(values["strain_rate"]),
            ])

            for _ in range(beads):
                if not f.readline():
                    raise ValueError(f"Incomplete XYZ frame: {filepath}")

    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError(f"Need at least three BD frames: {filepath}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Nonfinite BD values: {filepath}")
    if np.any(data[:, 1] <= 0):
        raise ValueError(f"Nonpositive diameter: {filepath}")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"BD times are not strictly increasing: {filepath}")

    return data[:, 0], data[:, 1], data[:, 2]


def crossover_time(time, rate):
    peaks = np.where(
        (rate[1:-1] > rate[:-2]) &
        (rate[1:-1] > rate[2:])
    )[0] + 1

    if len(peaks) == 0:
        raise ValueError(
            "No local strain-rate maximum found. Inspect the strain-rate "
            "curve and supply TC_OVERRIDE if appropriate."
        )

    return time[peaks[0]]


# ---------- READ EXPERIMENTAL DATA ----------

def experimental_pair(table, time_column, diameter_column):
    pair = table.iloc[:, [time_column, diameter_column]]
    pair = pair.apply(pd.to_numeric, errors="coerce")
    pair = pair.dropna()

    time = pair.iloc[:, 0].to_numpy(dtype=float)
    diameter_mm = pair.iloc[:, 1].to_numpy(dtype=float)

    keep = (
        np.isfinite(time) &
        np.isfinite(diameter_mm) &
        (diameter_mm > 0)
    )

    # MATLAB experimental diameters are in mm.
    return time[keep], diameter_mm[keep] / 2


# ---------- PLOT ONE PANEL ----------

def plot_panel(ax, species, concentrations, bd_files, experiment):
    if species == "PS16":
        colors = plt.cm.winter(
            np.linspace(0, 1, len(concentrations) + 1)
        )[:-1]
    else:
        colors = plt.cm.copper(
            np.linspace(0, 1, len(concentrations) + 1)
        )[:-1][::-1]

    for concentration, color in zip(concentrations, colors):
        exp_time, exp_radius = experiment[concentration]

        ax.plot(
            exp_time, exp_radius,
            "s:", color=color, markersize=4, markerfacecolor="none", linewidth=0.8,
            markeredgewidth=0.6, label=rf"$c_{{\mathrm{{{species}}}}}={concentration}$ [wppm]"
        )

        if concentration not in bd_files:
            continue

        time, diameter_m, rate = read_bd(bd_files[concentration])
        key = (species, concentration)

        if key in TC_OVERRIDE:
            tc = TC_OVERRIDE[key]
        else:
            tc = crossover_time(time, rate)

        radius_mm = diameter_m * 1000 / 2

        ax.plot(
            time - tc, radius_mm,
            "-", color=color, linewidth=2,
            # label=rf"$c_{{\mathrm{{{species}}}}}={concentration}$ [wppm], BD"
        )

        print(
            f"{species}, {concentration} wppm: "
            f"{len(time)} frames, tc={tc:.8g} s, "
            f"final t-tc={time[-1]-tc:.8g} s"
        )

    ax.set_yscale("log")
    ax.set_xlim(-0.05, 0.65)
    ax.set_ylim(2.5e-3, 0.5025)
    ax.set_xlabel(r"$t-t_c$ [s]")
    ax.set_ylabel(r"$a$ [mm]")

    other_species = "PS7" if species == "PS16" else "PS16"

    ax.text(
        0.90, 0.96,
        rf"$c_{{\mathrm{{{other_species}}}}}=0$ [wppm]",
        transform=ax.transAxes, fontsize=10,
        ha="right", va="top"
    )

    ax.legend(
        loc="upper right", bbox_to_anchor=(0.96, 0.90),
        frameon=False, fontsize=9,
        handlelength=2.5, labelspacing=0.35
    )


# ---------- LOAD THE SAME SHEETS/COLUMNS AS MATLAB ----------

with pd.ExcelFile(EXPERIMENT_FILE) as workbook:
    ps16_table = pd.read_excel(
        workbook, sheet_name="PS16", header=None
    )

    # MATLAB columns: 1:2, 5:6, 9:10, 13:14, 17:18, 21:22.
    # Python column indices start at zero.
    experiment_ps16 = {
        concentration: experimental_pair(ps16_table, 4*i, 4*i + 1)
        for i, concentration in enumerate([5, 10, 20, 50, 100, 200])
    }

    experiment_ps7 = {}
    for concentration in [200, 500, 1000]:
        table = pd.read_excel(
            workbook,
            sheet_name=f"{concentration}ppmPS7+PS16",
            header=None
        )
        experiment_ps7[concentration] = experimental_pair(table, 0, 1)


# ---------- FIGURE 4(a) AND 4(b) STYLE PANELS ----------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

plot_panel(
    axes[0], "PS16", [5, 10, 20, 50, 100, 200],
    BD_PS16, experiment_ps16
)
# axes[0].set_title("(a) PS16; PS7 = 0 wppm")

plot_panel(
    axes[1], "PS7", [200, 500, 1000],
    BD_PS7, experiment_ps7
)
# axes[1].set_title("(b) PS7; PS16 = 0 wppm")

fig.tight_layout()
plt.show()

# Optional: uncomment to save the figure.
fig.savefig("PS16_PS7_BD (FENE)_vs_experiment.png", dpi=300, bbox_inches="tight")
