import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

title = "mpi_o3"
write_results = False


def read_input_file(filename):
    variables = {}
    with open(filename, "r") as file:
        for line in file:
            key, value = line.strip().split("=", 1)
            key = key.strip()
            value = eval(value.strip())
            variables[key] = value
    return variables


save_path = "./graph/"
read_path = "./results/"

data = read_input_file(read_path + title + ".txt")
results_SMALL = data["results_SMALL"]
results_MIDDLE = data["results_MIDDLE"]
results_LARGE = data["results_LARGE"]
results_EXTLARGE = data["results_EXTLARGE"]

# write_results = True
# results_SMALL=[0.110787, 0.067214, 0.063037, 0.058088, 0.083459, 0.081243, 0.051404, 0.060263, 0.05155, 0.071098, 0.032519, 0.04515, 0.065723, 0.091094, 0.136212, 0.120807, 0.646959, 0.496774]
# results_MIDDLE=[0.931851, 0.497245, 0.437693, 0.540993, 0.566705, 0.669349, 0.566349, 0.337128, 0.358836, 0.304021, 0.142427, 0.198656, 0.292256, 0.376539, 0.624868, 0.428697, 0.941634, 0.677605]
# results_LARGE=[25.026308, 15.443051, 11.758131, 7.866836, 5.077578, 3.150042, 3.039332, 2.587785, 2.005004, 2.034282, 1.736706, 1.951426, 2.051116, 4.1813, 4.062121, 4.145007, 4.909373, 5.309015]
# results_EXTLARGE=[149.600611, 82.496043, 56.75903, 20.610923, 25.270447, 22.692715, 19.249695, 26.634133, 13.210994, 12.018836, 11.722738, 11.896462, 14.688057, 24.324574, 28.012847, 26.452516, 26.318955, 28.805096]

pipes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100, 120, 140, 160]
macros = ["SMALL", "MIDDLE", "LARGE", "EXTLARGE"]
lim = 350 if title == "mpi" else 150

all_results = [results_SMALL, results_MIDDLE, results_LARGE, results_EXTLARGE]
print(
    len(results_SMALL), len(results_MIDDLE), len(results_LARGE), len(results_EXTLARGE)
)

if write_results:
    with open(f"{read_path + title}.txt", "w") as file:
        print(f"{results_SMALL=}", file=file)
        print(f"{results_MIDDLE=}", file=file)
        print(f"{results_LARGE=}", file=file)
        print(f"{results_EXTLARGE=}", file=file)

print(f"{pipes=}")
print(f"{macros=}")
print("results_SMALL: ", [round(x, 2) for x in results_SMALL])
print("results_MIDDLE: ", [round(x, 2) for x in results_MIDDLE])
print("results_LARGE: ", [round(x, 2) for x in results_LARGE])
print("results_EXTLARGE: ", [round(x, 2) for x in results_EXTLARGE])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

X, Y = np.meshgrid(pipes, range(len(macros)))

Z = np.array([result for result in all_results])

surf = ax.plot_surface(X, Y, Z, cmap="inferno", edgecolor="none")

ax.set_xlabel("Pipes", fontsize=15, labelpad=12)
ax.set_ylabel("Macros", fontsize=15, labelpad=25)
ax.set_zlabel("Time", fontsize=15, labelpad=15)
ax.tick_params(axis="x", labelsize=13)
ax.tick_params(axis="y", pad=9, labelsize=13)
ax.tick_params(axis="z", labelsize=13)

ax.set_zlim([0, lim])

ax.set_yticks(range(len(macros)))
ax.set_yticklabels(macros)

cbar = fig.colorbar(surf, ax=ax, pad=0.15, shrink=0.6)

filename = f"{title}.pdf"
plt.savefig(save_path + filename, dpi=300, format="pdf")

macros_indices = range(len(macros))

fig, ax = plt.subplots(figsize=(12, 8))

for i, pipe in enumerate(pipes):
    line_data = [all_results[j][i] for j in range(len(all_results))]
    ax.plot(macros_indices, line_data, marker="o", label=f"Pipe: {pipe}")

ax.set_xticks(macros_indices)
ax.set_xticklabels(macros)
ax.set_xlabel("Macros", fontsize=20)
ax.set_ylabel("Time", fontsize=20)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_ylim([-10, lim])
ax.legend(fontsize=16.5)

plt.grid(True)
plt.tight_layout()

filename = f"{title}1.pdf"
plt.savefig(save_path + filename, dpi=300, format="pdf")

plt.show()
