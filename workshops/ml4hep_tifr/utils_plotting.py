"""Plotting helpers for the ML4HEP TIFR normalizing-flow tutorial.

The notebook remains responsible for sampling events and evaluating learned or
analytic probabilities.  Functions here turn those prepared arrays into
figures and can export completed figures as self-contained Python scripts,
keeping the pedagogical data flow visible in the exercise.
"""

from __future__ import annotations

import base64
import io
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import Collection, LineCollection, PathCollection
from matplotlib.contour import ContourSet
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, StepPatch
from matplotlib.text import Text
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2, norm


__all__ = [
    "export_standalone_figure_script",
    "plot_flow_pair_closure",
    "plot_log_density_truth_binned",
    "plot_log_density_truth_scatter",
    "plot_log_prob_cdf_closure",
    "plot_log_prob_closure",
    "plot_mu_hat_toys",
    "plot_profile_scan",
    "plot_profile_scan_comparison",
    "plot_t_mu_toys",
]


def _python_value(value):
    """Convert NumPy/Matplotlib scalar containers to plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_python_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        converted = [_python_value(item) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    return value


def _public_artist_label(artist):
    label = artist.get_label()
    if not isinstance(label, str) or not label or label.startswith("_"):
        return None
    return label


def _rgba_literal(color):
    """Return a serializable color value without depending on private APIs."""
    if isinstance(color, str):
        return color
    array = np.asarray(color)
    if array.ndim == 1:
        return tuple(float(value) for value in array)
    return [tuple(float(value) for value in row) for row in array]


def _line_style_kwargs(line):
    kwargs = {
        "color": _rgba_literal(line.get_color()),
        "linewidth": float(line.get_linewidth()),
        "linestyle": _python_value(line.get_linestyle()),
        "drawstyle": line.get_drawstyle(),
        "zorder": float(line.get_zorder()),
    }
    if line.get_alpha() is not None:
        kwargs["alpha"] = float(line.get_alpha())
    marker = line.get_marker()
    if marker not in (None, "None", "", " "):
        kwargs.update(
            {
                "marker": marker,
                "markersize": float(line.get_markersize()),
                "markerfacecolor": _rgba_literal(line.get_markerfacecolor()),
                "markeredgecolor": _rgba_literal(line.get_markeredgecolor()),
                "markeredgewidth": float(line.get_markeredgewidth()),
            }
        )
    label = _public_artist_label(line)
    if label is not None:
        kwargs["label"] = label
    return kwargs


def _patch_style_kwargs(patch):
    kwargs = {
        "facecolor": _rgba_literal(patch.get_facecolor()),
        "edgecolor": _rgba_literal(patch.get_edgecolor()),
        "linewidth": float(patch.get_linewidth()),
        "linestyle": _python_value(patch.get_linestyle()),
        "fill": bool(patch.get_fill()),
        "zorder": float(patch.get_zorder()),
    }
    if patch.get_alpha() is not None:
        kwargs["alpha"] = float(patch.get_alpha())
    if patch.get_hatch() is not None:
        kwargs["hatch"] = patch.get_hatch()
    label = _public_artist_label(patch)
    if label is not None:
        kwargs["label"] = label
    return kwargs


def _same_transform(left, right):
    if left is right:
        return True
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _weighted_correlation(values, weights=None):
    """Return a feature correlation matrix with optional event weights."""
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return np.corrcoef(values, rowvar=False)

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(values):
        raise ValueError("Correlation weights must match the event array.")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("Correlation weights must be finite and non-negative.")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("Correlation weights must have positive total weight.")

    normalized = weights / weight_sum
    mean = np.sum(normalized[:, None] * values, axis=0)
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    return correlation


def export_standalone_figure_script(
    fig,
    script_name,
    output_dir="exercise5_figures_scripts",
):
    """Write an editable, self-contained Python reconstruction of ``fig``.

    The generated script depends only on NumPy and Matplotlib.  All numerical
    artist data are stored in a compressed base85-encoded NPZ payload inside
    the script itself; no notebook variables, data files, or tutorial modules
    are needed when it is run later.  The visible Matplotlib construction is
    emitted as ordinary ``plot``, ``scatter``, ``stairs``, patch, and
    collection commands so labels, styles, ranges, and annotations remain easy
    to fine-tune.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Fully constructed figure to export.  Call this before clearing it.
    script_name : str or pathlib.Path
        Output filename.  Unsafe filename characters are replaced by ``_`` and
        the ``.py`` suffix is added when omitted.
    output_dir : str or pathlib.Path
        Directory receiving the generated script.

    Returns
    -------
    pathlib.Path
        Path of the written standalone script.
    """
    if fig is None or not hasattr(fig, "axes"):
        raise TypeError("fig must be a Matplotlib Figure.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_name = Path(script_name).name
    stem = Path(requested_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not safe_stem:
        raise ValueError("script_name must contain at least one safe character.")
    output_path = output_dir / f"{safe_stem}.py"

    # Finalize lazy layout/tick calculations before recording axes positions.
    try:
        fig.canvas.draw()
    except Exception:
        pass

    arrays = {}
    array_counter = 0

    def store_array(value, label="array"):
        nonlocal array_counter
        key = f"{label}_{array_counter:04d}"
        array_counter += 1
        array = np.ma.asarray(value)
        if np.ma.isMaskedArray(array):
            array = array.filled(np.nan)
        arrays[key] = np.asarray(array)
        return f'DATA[{key!r}]'

    lines = [
        '"""Standalone reconstruction generated from Exercise 5.\n\n',
        "The compressed DATA_B85 payload contains every numerical array used by\n",
        "the Matplotlib artists.  Edit the plotting commands or the clearly\n",
        "marked fine-tuning block near the bottom as needed.\n",
        '"""',
        "",
        "import base64",
        "import io",
        "from pathlib import Path",
        "",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "from matplotlib.collections import LineCollection, PolyCollection",
        "from matplotlib.path import Path as MplPath",
        "from matplotlib.patches import PathPatch, Polygon, Rectangle",
        "",
        "",
        "# DATA_B85 is filled below after all artist arrays have been collected.",
        "__DATA_PAYLOAD_PLACEHOLDER__",
        "",
        "with np.load(io.BytesIO(base64.b85decode(DATA_B85.encode('ascii')))) as payload:",
        "    DATA = {name: payload[name] for name in payload.files}",
        "",
        "",
        f"fig = plt.figure(figsize={tuple(float(v) for v in fig.get_size_inches())!r}, dpi={float(fig.dpi)!r})",
        f"fig.patch.set_facecolor({_rgba_literal(fig.get_facecolor())!r})",
        "axes = []",
    ]

    axis_legend_specs = []
    for axis_index, ax in enumerate(fig.axes):
        axis_name = f"ax_{axis_index}"
        bounds = tuple(float(value) for value in ax.get_position().bounds)
        lines.extend(
            [
                "",
                f"# Axes {axis_index}",
                f"{axis_name} = fig.add_axes({bounds!r})",
                f"axes.append({axis_name})",
                f"{axis_name}.set_facecolor({_rgba_literal(ax.get_facecolor())!r})",
            ]
        )

        for artist in ax.get_children():
            if artist is ax.patch or not artist.get_visible():
                continue

            if isinstance(artist, Line2D):
                x_data = np.asarray(artist.get_xdata(orig=False))
                y_data = np.asarray(artist.get_ydata(orig=False))
                if x_data.ndim != 1 or y_data.ndim != 1 or len(x_data) != len(y_data):
                    continue
                kwargs = _line_style_kwargs(artist)
                x_ref = store_array(x_data, f"ax{axis_index}_line_x")
                y_ref = store_array(y_data, f"ax{axis_index}_line_y")
                transform = artist.get_transform()
                if (
                    len(x_data) >= 2
                    and np.allclose(x_data, x_data[0], equal_nan=True)
                    and _same_transform(transform, ax.get_xaxis_transform())
                ):
                    lines.append(
                        f"{axis_name}.axvline({float(x_data[0])!r}, **{kwargs!r})"
                    )
                elif (
                    len(y_data) >= 2
                    and np.allclose(y_data, y_data[0], equal_nan=True)
                    and _same_transform(transform, ax.get_yaxis_transform())
                ):
                    lines.append(
                        f"{axis_name}.axhline({float(y_data[0])!r}, **{kwargs!r})"
                    )
                else:
                    lines.append(f"{axis_name}.plot({x_ref}, {y_ref}, **{kwargs!r})")

            elif isinstance(artist, ContourSet):
                paths = artist.get_paths()
                edgecolors = artist.get_edgecolors()
                linewidths = artist.get_linewidths()
                linestyles = artist.get_linestyles()
                label = _public_artist_label(artist)
                for path_index, path in enumerate(paths):
                    vertices_ref = store_array(
                        path.vertices, f"ax{axis_index}_contour_vertices"
                    )
                    if path.codes is None:
                        path_expression = f"MplPath({vertices_ref})"
                    else:
                        codes_ref = store_array(
                            path.codes, f"ax{axis_index}_contour_codes"
                        )
                        path_expression = f"MplPath({vertices_ref}, {codes_ref})"
                    color = (
                        tuple(float(v) for v in edgecolors[path_index % len(edgecolors)])
                        if len(edgecolors)
                        else "black"
                    )
                    linewidth = (
                        float(linewidths[path_index % len(linewidths)])
                        if len(linewidths)
                        else 1.0
                    )
                    linestyle = (
                        _python_value(linestyles[path_index % len(linestyles)])
                        if len(linestyles)
                        else "solid"
                    )
                    kwargs = {
                        "facecolor": "none",
                        "edgecolor": color,
                        "linewidth": linewidth,
                        "linestyle": linestyle,
                        "fill": False,
                        "zorder": float(artist.get_zorder()),
                    }
                    if label is not None and path_index == 0:
                        kwargs["label"] = label
                    lines.extend(
                        [
                            f"_patch = PathPatch({path_expression}, **{kwargs!r})",
                            f"{axis_name}.add_patch(_patch)",
                        ]
                    )

            elif isinstance(artist, PathCollection):
                offsets = np.asarray(artist.get_offsets())
                if offsets.ndim != 2 or offsets.shape[1] != 2:
                    continue
                offsets_ref = store_array(offsets, f"ax{axis_index}_scatter_offsets")
                sizes_ref = store_array(artist.get_sizes(), f"ax{axis_index}_scatter_sizes")
                kwargs = {
                    "s": f"__ARRAY__:{sizes_ref}",
                    "linewidths": _python_value(artist.get_linewidths()),
                    "zorder": float(artist.get_zorder()),
                }
                facecolors = artist.get_facecolors()
                edgecolors = artist.get_edgecolors()
                if len(facecolors):
                    face_ref = store_array(
                        facecolors, f"ax{axis_index}_scatter_facecolors"
                    )
                    kwargs["facecolors"] = f"__ARRAY__:{face_ref}"
                if len(edgecolors):
                    edge_ref = store_array(
                        edgecolors, f"ax{axis_index}_scatter_edgecolors"
                    )
                    kwargs["edgecolors"] = f"__ARRAY__:{edge_ref}"
                paths = artist.get_paths()
                marker_setup = []
                if paths:
                    marker_vertices = store_array(
                        paths[0].vertices, f"ax{axis_index}_marker_vertices"
                    )
                    if paths[0].codes is None:
                        marker_expression = f"MplPath({marker_vertices})"
                    else:
                        marker_codes = store_array(
                            paths[0].codes, f"ax{axis_index}_marker_codes"
                        )
                        marker_expression = f"MplPath({marker_vertices}, {marker_codes})"
                    marker_setup.append(f"_marker = {marker_expression}")
                    kwargs["marker"] = "__EXPR__:_marker"
                if artist.get_alpha() is not None:
                    kwargs["alpha"] = float(artist.get_alpha())
                label = _public_artist_label(artist)
                if label is not None:
                    kwargs["label"] = label

                rendered_kwargs = []
                for key, value in kwargs.items():
                    if isinstance(value, str) and value.startswith("__ARRAY__:"):
                        rendered_kwargs.append(f"{key}={value.removeprefix('__ARRAY__:')}")
                    elif isinstance(value, str) and value.startswith("__EXPR__:"):
                        rendered_kwargs.append(f"{key}={value.removeprefix('__EXPR__:')}")
                    else:
                        rendered_kwargs.append(f"{key}={value!r}")
                lines.extend(marker_setup)
                lines.append(
                    f"{axis_name}.scatter({offsets_ref}[:, 0], {offsets_ref}[:, 1], "
                    + ", ".join(rendered_kwargs)
                    + ")"
                )

            elif isinstance(artist, LineCollection):
                valid_segments = []
                for segment in artist.get_segments():
                    segment = np.asarray(segment)
                    if segment.ndim == 2 and segment.shape[1] == 2 and len(segment):
                        valid_segments.append(segment)
                if not valid_segments:
                    continue
                segment_refs = [
                    store_array(segment, f"ax{axis_index}_line_segment")
                    for segment in valid_segments
                ]
                kwargs = {
                    "colors": _rgba_literal(artist.get_edgecolors()),
                    "linewidths": _python_value(artist.get_linewidths()),
                    "linestyles": _python_value(artist.get_linestyles()),
                    "zorder": float(artist.get_zorder()),
                }
                if artist.get_alpha() is not None:
                    kwargs["alpha"] = float(artist.get_alpha())
                label = _public_artist_label(artist)
                if label is not None:
                    kwargs["label"] = label
                lines.extend(
                    [
                        f"_collection = LineCollection([{', '.join(segment_refs)}], **{kwargs!r})",
                        f"{axis_name}.add_collection(_collection)",
                    ]
                )

            elif isinstance(artist, Collection):
                # Fill-between and other polygonal collections are reproduced
                # from their fully resolved paths, requiring no source data.
                path_refs = []
                for path in artist.get_paths():
                    vertices_ref = store_array(
                        path.vertices, f"ax{axis_index}_poly_vertices"
                    )
                    path_refs.append(vertices_ref)
                if not path_refs:
                    continue
                kwargs = {
                    "facecolors": _rgba_literal(artist.get_facecolors()),
                    "edgecolors": _rgba_literal(artist.get_edgecolors()),
                    "linewidths": _python_value(artist.get_linewidths()),
                    "linestyles": _python_value(artist.get_linestyles()),
                    "zorder": float(artist.get_zorder()),
                }
                if artist.get_alpha() is not None:
                    kwargs["alpha"] = float(artist.get_alpha())
                label = _public_artist_label(artist)
                if label is not None:
                    kwargs["label"] = label
                lines.extend(
                    [
                        f"_collection = PolyCollection([{', '.join(path_refs)}], **{kwargs!r})",
                        f"{axis_name}.add_collection(_collection)",
                    ]
                )

            elif isinstance(artist, StepPatch):
                stair_data = artist.get_data()
                values_ref = store_array(
                    stair_data.values, f"ax{axis_index}_stairs_values"
                )
                edges_ref = store_array(
                    stair_data.edges, f"ax{axis_index}_stairs_edges"
                )
                baseline = np.asarray(stair_data.baseline)
                baseline_value = (
                    float(baseline)
                    if baseline.ndim == 0
                    else f"__ARRAY__:{store_array(baseline, f'ax{axis_index}_stairs_baseline')}"
                )
                kwargs = _patch_style_kwargs(artist)
                kwargs["orientation"] = getattr(artist, "_orientation", "vertical")
                rendered_kwargs = []
                for key, value in kwargs.items():
                    rendered_kwargs.append(f"{key}={value!r}")
                baseline_expression = (
                    baseline_value.removeprefix("__ARRAY__:")
                    if isinstance(baseline_value, str)
                    and baseline_value.startswith("__ARRAY__:")
                    else repr(baseline_value)
                )
                lines.append(
                    f"{axis_name}.stairs({values_ref}, {edges_ref}, "
                    f"baseline={baseline_expression}, "
                    + ", ".join(rendered_kwargs)
                    + ")"
                )

            elif isinstance(artist, Polygon):
                xy_ref = store_array(artist.get_xy(), f"ax{axis_index}_polygon")
                kwargs = _patch_style_kwargs(artist)
                lines.extend(
                    [
                        f"_patch = Polygon({xy_ref}, closed={bool(artist.get_closed())!r}, **{kwargs!r})",
                        f"{axis_name}.add_patch(_patch)",
                    ]
                )

            elif isinstance(artist, Rectangle):
                x, y = artist.get_xy()
                kwargs = _patch_style_kwargs(artist)
                lines.extend(
                    [
                        f"_patch = Rectangle(({float(x)!r}, {float(y)!r}), {float(artist.get_width())!r}, {float(artist.get_height())!r}, angle={float(artist.angle)!r}, **{kwargs!r})",
                        f"{axis_name}.add_patch(_patch)",
                    ]
                )

            elif isinstance(artist, Text) and artist not in {
                ax.title,
                ax.xaxis.label,
                ax.yaxis.label,
            }:
                text = artist.get_text()
                if not text:
                    continue
                transform_expression = (
                    f"{axis_name}.transAxes"
                    if _same_transform(artist.get_transform(), ax.transAxes)
                    else f"{axis_name}.transData"
                )
                x, y = artist.get_position()
                kwargs = {
                    "transform": f"__EXPR__:{transform_expression}",
                    "ha": artist.get_horizontalalignment(),
                    "va": artist.get_verticalalignment(),
                    "fontsize": float(artist.get_fontsize()),
                    "fontweight": artist.get_fontweight(),
                    "fontstyle": artist.get_fontstyle(),
                    "color": _rgba_literal(artist.get_color()),
                    "rotation": float(artist.get_rotation()),
                    "zorder": float(artist.get_zorder()),
                }
                if artist.get_alpha() is not None:
                    kwargs["alpha"] = float(artist.get_alpha())
                rendered_kwargs = []
                for key, value in kwargs.items():
                    if isinstance(value, str) and value.startswith("__EXPR__:"):
                        rendered_kwargs.append(f"{key}={value.removeprefix('__EXPR__:')}")
                    else:
                        rendered_kwargs.append(f"{key}={value!r}")
                lines.append(
                    f"{axis_name}.text({float(x)!r}, {float(y)!r}, {text!r}, "
                    + ", ".join(rendered_kwargs)
                    + ")"
                )

        x_scale = ax.get_xscale()
        y_scale = ax.get_yscale()
        if x_scale != "linear":
            lines.append(f"{axis_name}.set_xscale({x_scale!r})")
        if y_scale != "linear":
            lines.append(f"{axis_name}.set_yscale({y_scale!r})")
        lines.extend(
            [
                f"{axis_name}.set_xlim({tuple(float(v) for v in ax.get_xlim())!r})",
                f"{axis_name}.set_ylim({tuple(float(v) for v in ax.get_ylim())!r})",
            ]
        )
        if ax.get_xlabel():
            lines.append(
                f"{axis_name}.set_xlabel({ax.get_xlabel()!r}, fontsize={float(ax.xaxis.label.get_fontsize())!r})"
            )
        if ax.get_ylabel():
            lines.append(
                f"{axis_name}.set_ylabel({ax.get_ylabel()!r}, fontsize={float(ax.yaxis.label.get_fontsize())!r})"
            )
        if ax.get_title():
            lines.append(
                f"{axis_name}.set_title({ax.get_title()!r}, fontsize={float(ax.title.get_fontsize())!r})"
            )

        x_ticklabels = ax.get_xticklabels()
        y_ticklabels = ax.get_yticklabels()
        x_labels_visible = any(label.get_visible() for label in x_ticklabels)
        y_labels_visible = any(label.get_visible() for label in y_ticklabels)
        x_fontsize = next(
            (float(label.get_fontsize()) for label in x_ticklabels if label.get_visible()),
            None,
        )
        y_fontsize = next(
            (float(label.get_fontsize()) for label in y_ticklabels if label.get_visible()),
            None,
        )
        lines.append(
            f"{axis_name}.tick_params(axis='x', labelbottom={x_labels_visible!r}"
            + (f", labelsize={x_fontsize!r}" if x_fontsize is not None else "")
            + ")"
        )
        lines.append(
            f"{axis_name}.tick_params(axis='y', labelleft={y_labels_visible!r}"
            + (f", labelsize={y_fontsize!r}" if y_fontsize is not None else "")
            + ")"
        )

        x_gridlines = [line for line in ax.get_xgridlines() if line.get_visible()]
        y_gridlines = [line for line in ax.get_ygridlines() if line.get_visible()]
        if x_gridlines or y_gridlines:
            exemplar = (x_gridlines or y_gridlines)[0]
            grid_axis = "both" if x_gridlines and y_gridlines else ("x" if x_gridlines else "y")
            grid_kwargs = {
                "axis": grid_axis,
                "color": _rgba_literal(exemplar.get_color()),
                "linestyle": exemplar.get_linestyle(),
                "linewidth": float(exemplar.get_linewidth()),
            }
            if exemplar.get_alpha() is not None:
                grid_kwargs["alpha"] = float(exemplar.get_alpha())
            lines.append(f"{axis_name}.grid(**{grid_kwargs!r})")

        if not ax.axison:
            lines.append(f"{axis_name}.set_axis_off()")

        for spine_name, spine in ax.spines.items():
            if not spine.get_visible():
                lines.append(f"{axis_name}.spines[{spine_name!r}].set_visible(False)")
            else:
                lines.append(
                    f"{axis_name}.spines[{spine_name!r}].set_linewidth({float(spine.get_linewidth())!r})"
                )

        legend = ax.get_legend()
        if legend is not None:
            fontsize = (
                float(legend.get_texts()[0].get_fontsize())
                if legend.get_texts()
                else None
            )
            axis_legend_specs.append(
                (
                    axis_name,
                    {
                        "loc": _python_value(getattr(legend, "_loc", "best")),
                        "frameon": bool(legend.get_frame_on()),
                        "ncols": int(getattr(legend, "_ncols", 1)),
                        **({"fontsize": fontsize} if fontsize is not None else {}),
                    },
                )
            )

    for axis_name, kwargs in axis_legend_specs:
        lines.append(f"{axis_name}.legend(**{kwargs!r})")

    for legend in fig.legends:
        desired_labels = [text.get_text() for text in legend.get_texts()]
        bbox = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds
        bbox = tuple(float(value) for value in bbox)
        bbox_anchor = bbox[:2] if np.allclose(bbox[2:], 0.0) else bbox
        fontsize = (
            float(legend.get_texts()[0].get_fontsize())
            if legend.get_texts()
            else None
        )
        kwargs = {
            "loc": _python_value(getattr(legend, "_loc", "best")),
            "bbox_to_anchor": bbox_anchor,
            "ncols": int(getattr(legend, "_ncols", 1)),
            "frameon": bool(legend.get_frame_on()),
            **({"fontsize": fontsize} if fontsize is not None else {}),
        }
        lines.append(
            "_handles, _labels = [], []\n"
            "for _ax in axes:\n"
            "    _h, _l = _ax.get_legend_handles_labels()\n"
            "    _handles.extend(_h)\n"
            "    _labels.extend(_l)\n"
            "_handle_by_label = dict(zip(_labels, _handles))\n"
            f"_desired_labels = {desired_labels!r}\n"
            "_selected_labels = [label for label in _desired_labels if label in _handle_by_label]\n"
            "_selected_handles = [_handle_by_label[label] for label in _selected_labels]\n"
            f"fig.legend(_selected_handles, _selected_labels, **{kwargs!r})"
        )

    if getattr(fig, "_suptitle", None) is not None:
        suptitle = fig._suptitle
        x, y = suptitle.get_position()
        lines.append(
            f"fig.suptitle({suptitle.get_text()!r}, x={float(x)!r}, y={float(y)!r}, "
            f"fontsize={float(suptitle.get_fontsize())!r}, "
            f"fontweight={suptitle.get_fontweight()!r})"
        )

    lines.extend(
        [
            "",
            "# -------------------------------------------------------------------------",
            "# Fine-tuning hook: edit any artist through `fig` or `axes` here.",
            "# Example: axes[0].set_title('My revised title')",
            "# -------------------------------------------------------------------------",
            "",
            "output_path = Path(__file__).with_suffix('.png')",
            "fig.savefig(output_path, bbox_inches='tight')",
            "print(f'Wrote {output_path}')",
            "plt.show()",
            "",
        ]
    )

    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    encoded = base64.b85encode(payload.getvalue()).decode("ascii")
    payload_lines = ["DATA_B85 = ("]
    payload_lines.extend(f"    {chunk!r}" for chunk in textwrap.wrap(encoded, 100))
    payload_lines.append(")")
    payload_source = "\n".join(payload_lines)
    source = "\n".join(lines).replace(
        "__DATA_PAYLOAD_PLACEHOLDER__", payload_source
    )
    output_path.write_text(source, encoding="utf-8")
    print(f"Wrote standalone figure script: {output_path}")
    return output_path


def _highest_density_contour_levels(histogram, probabilities=(0.95, 0.68)):
    """Return density thresholds enclosing the requested probability masses."""
    values = np.asarray(histogram, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) == 0:
        return np.array([])

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    levels = []
    for probability in probabilities:
        index = min(np.searchsorted(cumulative, probability), len(ordered) - 1)
        levels.append(ordered[index])

    levels = np.unique(np.sort(levels))
    return levels[(levels > 0.0) & (levels < ordered[0])]


def plot_flow_pair_closure(
    sample_name,
    mc,
    generated,
    feature_names,
    n_bins_1d=60,
    n_bins_2d=35,
    contour_smoothing=1.0,
    quantile_range=(0.005, 0.995),
    mc_weights=None,
    generated_weights=None,
    mc_label="held-out MC",
    generated_label="flow sample",
    generated_color="C1",
    correlation_names=("MC", "flow"),
):
    """Pair plot of two event arrays, optionally with event weights."""
    mc = np.asarray(mc)
    generated = np.asarray(generated)
    feature_names = list(feature_names)
    if mc.ndim != 2 or generated.ndim != 2 or mc.shape[1] != generated.shape[1]:
        raise ValueError("mc and generated must be 2D arrays with matching columns.")
    if mc.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the number of array columns.")
    if len(correlation_names) != 2:
        raise ValueError("correlation_names must contain exactly two labels.")

    if mc_weights is not None:
        mc_weights = np.asarray(mc_weights, dtype=np.float64).reshape(-1)
        if len(mc_weights) != len(mc):
            raise ValueError("mc_weights must match the held-out event array.")
    if generated_weights is not None:
        generated_weights = np.asarray(
            generated_weights, dtype=np.float64
        ).reshape(-1)
        if len(generated_weights) != len(generated):
            raise ValueError(
                "generated_weights must match the generated event array."
            )

    q_low, q_high = quantile_range
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("quantile_range must satisfy 0 <= low < high <= 1.")

    n_features = len(feature_names)
    limits = []
    for feature_index in range(n_features):
        low = min(
            np.quantile(mc[:, feature_index], q_low),
            np.quantile(generated[:, feature_index], q_low),
        )
        high = max(
            np.quantile(mc[:, feature_index], q_high),
            np.quantile(generated[:, feature_index], q_high),
        )
        limits.append((low, high))

    corr_mc = _weighted_correlation(mc, mc_weights)
    corr_generated = _weighted_correlation(generated, generated_weights)
    fig, axes = plt.subplots(
        n_features,
        n_features,
        figsize=(2.55 * n_features, 2.55 * n_features),
        squeeze=False,
    )

    for row in range(n_features):
        for column in range(n_features):
            ax = axes[row, column]
            x_low, x_high = limits[column]

            if row == column:
                bins = np.linspace(x_low, x_high, n_bins_1d + 1)
                ax.hist(
                    mc[:, column],
                    bins=bins,
                    weights=mc_weights,
                    density=True,
                    histtype="step",
                    color="black",
                    lw=1.8,
                    label=mc_label,
                )
                ax.hist(
                    generated[:, column],
                    bins=bins,
                    weights=generated_weights,
                    density=True,
                    histtype="step",
                    color=generated_color,
                    lw=1.8,
                    ls="--",
                    label=generated_label,
                )
                ax.set_xlim(x_low, x_high)
                if row == 0:
                    ax.set_ylabel("density")

            elif row > column:
                y_low, y_high = limits[row]
                x_edges = np.linspace(x_low, x_high, n_bins_2d + 1)
                y_edges = np.linspace(y_low, y_high, n_bins_2d + 1)
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

                hist_mc, _, _ = np.histogram2d(
                    mc[:, column],
                    mc[:, row],
                    bins=(x_edges, y_edges),
                    weights=mc_weights,
                )
                hist_generated, _, _ = np.histogram2d(
                    generated[:, column],
                    generated[:, row],
                    bins=(x_edges, y_edges),
                    weights=generated_weights,
                )
                if contour_smoothing > 0.0:
                    hist_mc = gaussian_filter(hist_mc, sigma=contour_smoothing)
                    hist_generated = gaussian_filter(
                        hist_generated, sigma=contour_smoothing
                    )

                levels_mc = _highest_density_contour_levels(hist_mc)
                levels_generated = _highest_density_contour_levels(hist_generated)
                if len(levels_mc):
                    ax.contour(
                        x_centers,
                        y_centers,
                        hist_mc.T,
                        levels=levels_mc,
                        colors="black",
                        linewidths=1.5,
                    )
                if len(levels_generated):
                    ax.contour(
                        x_centers,
                        y_centers,
                        hist_generated.T,
                        levels=levels_generated,
                        colors=generated_color,
                        linestyles="--",
                        linewidths=1.5,
                    )
                ax.set_xlim(x_low, x_high)
                ax.set_ylim(y_low, y_high)

            else:
                rho_mc = corr_mc[row, column]
                rho_generated = corr_generated[row, column]
                delta_rho = rho_generated - rho_mc
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.62,
                    rf"$\rho_{{\rm {correlation_names[0]}}}={rho_mc:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.text(
                    0.5,
                    0.44,
                    rf"$\rho_{{\rm {correlation_names[1]}}}={rho_generated:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color=generated_color,
                )
                ax.text(
                    0.5,
                    0.25,
                    rf"$\Delta\rho={delta_rho:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                )

            if row >= column:
                ax.tick_params(
                    axis="x",
                    labelbottom=(row == n_features - 1),
                    labelsize=8,
                )
                ax.tick_params(
                    axis="y",
                    labelleft=(column == 0),
                    labelsize=8,
                )
                if row == n_features - 1:
                    ax.set_xlabel(feature_names[column])
                if column == 0 and row > 0:
                    ax.set_ylabel(feature_names[row])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(f"{sample_name}: feature and correlation closure", y=0.999)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig


def plot_log_prob_closure(
    sample_name,
    log_p_mc,
    log_p_generated,
    n_bins=60,
    quantile_range=(0.001, 0.999),
    flow_color="C1",
):
    """Histogram and ratio for two prepared learned-log-density arrays."""
    log_p_mc = np.asarray(log_p_mc)
    log_p_generated = np.asarray(log_p_generated)
    if len(log_p_mc) == 0 or len(log_p_generated) == 0:
        raise ValueError("The log-probability arrays must be non-empty.")

    q_low, q_high = quantile_range
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("quantile_range must satisfy 0 <= low < high <= 1.")
    x_min = min(
        np.quantile(log_p_mc, q_low),
        np.quantile(log_p_generated, q_low),
    ) - 5.0
    x_max = max(
        np.quantile(log_p_mc, q_high),
        np.quantile(log_p_generated, q_high),
    ) + 5.0

    edges = np.linspace(x_min, x_max, n_bins + 1)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mc_counts, _ = np.histogram(log_p_mc, bins=edges)
    generated_counts, _ = np.histogram(log_p_generated, bins=edges)

    mc_norm = mc_counts.sum()
    generated_norm = generated_counts.sum()
    if mc_norm == 0 or generated_norm == 0:
        raise ValueError("No log-probability entries fall inside the plotting range.")
    mc_density = mc_counts / (mc_norm * widths)
    mc_error = np.sqrt(mc_counts) / (mc_norm * widths)
    generated_density = generated_counts / (generated_norm * widths)

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.stairs(
        generated_density,
        edges,
        color=flow_color,
        lw=2,
        fill=True,
        alpha=0.22,
        label=f"flow sample ({len(log_p_generated):,} events)",
    )
    ax.stairs(generated_density, edges, color=flow_color, lw=2)
    ax.errorbar(
        centers,
        mc_density,
        yerr=mc_error,
        fmt="o",
        ms=3.5,
        color="black",
        capsize=1.5,
        label=f"held-out MC ({len(log_p_mc):,} events)",
    )
    ax.set_ylabel("normalized events\n/ unit log density")
    ax.set_title(f"{sample_name}: log-density closure")
    ax.legend(fontsize=9)

    valid = (mc_counts > 0) & (generated_counts > 0)
    ratio = mc_density[valid] / generated_density[valid]
    ratio_error = ratio * np.sqrt(
        1.0 / mc_counts[valid] + 1.0 / generated_counts[valid]
    )
    ratio_ax.errorbar(
        centers[valid],
        ratio,
        yerr=ratio_error,
        fmt="o",
        ms=3.5,
        color="black",
        capsize=1.5,
    )
    ratio_ax.axhline(1.0, color=flow_color, lw=1.5)
    ratio_ax.set_xlabel(r"$\log \hat p_{\rm flow}(x)$")
    ratio_ax.set_ylabel("MC / flow")
    ratio_ax.set_xlim(edges[0], edges[-1])
    ratio_ax.grid(axis="y", alpha=0.25)
    fig.align_ylabels((ax, ratio_ax))
    fig.subplots_adjust(hspace=0.05)
    return fig


def plot_log_prob_cdf_closure(
    sample_name,
    log_p_mc,
    log_p_generated,
    n_cdf_points=200,
    cdf_range=(0.001, 0.999),
    color="C0",
):
    """P-P plot and CDF ratio for prepared learned-log-density arrays."""
    log_p_mc = np.asarray(log_p_mc)
    log_p_generated = np.asarray(log_p_generated)
    flow_cdf = np.linspace(cdf_range[0], cdf_range[1], n_cdf_points)
    if not 0.0 < flow_cdf[0] < flow_cdf[-1] < 1.0:
        raise ValueError("cdf_range must lie strictly inside (0, 1).")

    thresholds = np.quantile(log_p_generated, flow_cdf)
    sorted_mc = np.sort(log_p_mc)
    mc_cdf = np.searchsorted(sorted_mc, thresholds, side="right") / len(sorted_mc)
    cdf_sigma = np.sqrt(
        flow_cdf
        * (1.0 - flow_cdf)
        * (1.0 / len(log_p_mc) + 1.0 / len(log_p_generated))
    )
    lower = np.clip(flow_cdf - cdf_sigma, 0.0, 1.0)
    upper = np.clip(flow_cdf + cdf_sigma, 0.0, 1.0)

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(6.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.fill_between(
        flow_cdf,
        lower,
        upper,
        color=color,
        alpha=0.18,
        label=r"pointwise sampling expectation ($\pm1\sigma$)",
    )
    ax.plot(flow_cdf, mc_cdf, color=color, lw=2, label="held-out MC versus flow")
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="black",
        ls="--",
        lw=1.5,
        label="perfect closure",
    )
    ax.set_ylabel(r"MC CDF $F_{\rm MC}(\log \hat p)$")
    ax.set_title(f"{sample_name}: log-density CDF closure")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    cdf_ratio = mc_cdf / flow_cdf
    ratio_sigma = cdf_sigma / flow_cdf
    ratio_ax.fill_between(
        flow_cdf,
        np.clip(1.0 - ratio_sigma, 0.0, None),
        1.0 + ratio_sigma,
        color=color,
        alpha=0.18,
    )
    ratio_ax.plot(flow_cdf, cdf_ratio, color=color, lw=2)
    ratio_ax.axhline(1.0, color="black", ls="--", lw=1.5)
    ratio_ax.set_xlabel(r"flow CDF $F_{\rm flow}(\log \hat p)$")
    ratio_ax.set_ylabel(r"$F_{\rm MC}/F_{\rm flow}$")
    ratio_ax.set_xlim(0.0, 1.0)
    ratio_ax.grid(axis="y", alpha=0.25)

    ratio_span = max(0.05, 1.1 * np.max(np.abs(cdf_ratio - 1.0) + ratio_sigma))
    ratio_ax.set_ylim(max(0.0, 1.0 - ratio_span), 1.0 + ratio_span)
    fig.align_ylabels((ax, ratio_ax))
    fig.subplots_adjust(hspace=0.05)
    return fig


def plot_log_density_truth_scatter(sample_name, log_p_truth, log_p_flow):
    """Scatter plot of prepared analytic and learned log-density arrays."""
    log_p_truth = np.asarray(log_p_truth)
    log_p_flow = np.asarray(log_p_flow)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(log_p_truth, log_p_flow, s=4, alpha=0.25)
    lo = min(log_p_truth.min(), log_p_flow.min())
    hi = max(log_p_truth.max(), log_p_flow.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("truth log density")
    ax.set_ylabel("flow log density")
    ax.set_title(sample_name)
    fig.tight_layout()
    return fig


def plot_log_density_truth_binned(
    sample_name,
    calibration,
    edges,
    log_p_truth=None,
    log_p_flow=None,
    show_scatter=False,
):
    """Plot a prepared binned log-density calibration table."""
    edges = np.asarray(edges)
    valid = calibration["count"] > 0
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    if show_scatter:
        if log_p_truth is None or log_p_flow is None:
            raise ValueError("Raw log-density arrays are required for show_scatter=True.")
        ax.scatter(log_p_truth, log_p_flow, s=3, alpha=0.10)

    flow_step = calibration["flow_mean"].to_numpy()
    delta_step = calibration["delta_mean"].to_numpy()
    ax.step(edges, np.r_[flow_step, flow_step[-1]], where="post", lw=2, label="bin mean")
    ax.errorbar(
        calibration.loc[valid, "truth_mean"],
        calibration.loc[valid, "flow_mean"],
        yerr=calibration.loc[valid, "flow_sem"],
        fmt="o",
        ms=3,
        capsize=2,
    )
    lo = min(edges[0], np.nanmin(flow_step))
    hi = max(edges[-1], np.nanmax(flow_step))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect calibration")
    ax.set_ylabel(r"$\langle \log \hat p_{\rm flow} \rangle$")
    ax.set_title(f"{sample_name}: binned log-density calibration")
    ax.legend(fontsize=8)

    residual_ax.step(edges, np.r_[delta_step, delta_step[-1]], where="post", lw=2)
    residual_ax.errorbar(
        calibration.loc[valid, "truth_mean"],
        calibration.loc[valid, "delta_mean"],
        yerr=calibration.loc[valid, "delta_sem"],
        fmt="o",
        ms=3,
        capsize=2,
    )
    residual_ax.axhline(0.0, color="k", lw=1)
    residual_ax.set_xlabel(r"truth $\log p(x)$ bin")
    residual_ax.set_ylabel(r"$\langle \Delta \log p \rangle$")
    residual_ax.set_xlim(edges[0], edges[-1])
    return fig


def plot_profile_scan(scan, t_mu, label="Flow densities", reference_mu=1.0):
    """Plot one prepared profile-likelihood scan."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(scan, t_mu, lw=2, label=label)
    ax.axvline(reference_mu, color="black", ls=":", lw=1, alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\mu_{\rm signal}$")
    ax.set_ylabel(r"$t_\mu$")
    ax.legend()
    return fig


def plot_profile_scan_comparison(
    scan_flow,
    t_mu_flow,
    mu_min_flow,
    scan_truth,
    t_mu_truth,
    mu_min_truth,
    reference_mu=1.0,
):
    """Compare prepared learned- and analytic-density likelihood scans."""
    mu_min_flow = float(np.asarray(mu_min_flow).reshape(-1)[0])
    mu_min_truth = float(np.asarray(mu_min_truth).reshape(-1)[0])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        np.asarray(scan_flow) - mu_min_flow + reference_mu,
        t_mu_flow,
        lw=2,
        label="Flow densities",
    )
    ax.plot(
        np.asarray(scan_truth) - mu_min_truth + reference_mu,
        t_mu_truth,
        lw=2,
        ls="--",
        label="Analytic reco densities",
    )
    ax.axvline(reference_mu, color="black", ls=":", lw=1, alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\mu_{\rm signal}$")
    ax.set_ylabel(r"$t_\mu$")
    ax.legend()
    return fig


def plot_t_mu_toys(toy_results, n_bins=35):
    """Compare prepared toy test statistics with Wilks/Cowan expectations."""
    hypotheses = sorted(toy_results["mu_true"].unique())
    fig, axes = plt.subplots(1, len(hypotheses), figsize=(6.2 * len(hypotheses), 4.6))
    axes = np.atleast_1d(axes)

    for ax, mu_true in zip(axes, hypotheses):
        values = toy_results.loc[toy_results["mu_true"] == mu_true, "t_mu"].to_numpy()
        upper = max(float(chi2.ppf(0.999, df=1)), 1.001 * float(values.max()))
        edges = np.linspace(0.0, upper, n_bins + 1)
        toy_probability = np.histogram(values, bins=edges)[0] / len(values)
        chi2_probability = np.diff(chi2.cdf(edges, df=1))

        if np.isclose(mu_true, 0.0):
            asymptotic_probability = 0.5 * chi2_probability
            asymptotic_probability[0] += 0.5
            theory_label = r"Cowan: $\frac{1}{2}\delta(0)+\frac{1}{2}\chi^2_1$"
        else:
            asymptotic_probability = chi2_probability
            theory_label = r"Wilks: $\chi^2_1$"

        ax.stairs(
            toy_probability,
            edges,
            color="C0",
            lw=2,
            fill=True,
            alpha=0.22,
            label="flow toys",
        )
        ax.stairs(
            asymptotic_probability,
            edges,
            color="C3",
            lw=2,
            ls="--",
            label=theory_label,
        )
        ax.set_xlabel(r"$t_\mu$")
        ax.set_ylabel("probability / bin")
        ax.set_title(rf"toys generated at $\mu={mu_true:g}$")
        ax.legend(fontsize=9)
        if np.isclose(mu_true, 0.0):
            zero_fraction = np.mean(values < 1.0e-10)
            ax.text(
                0.97,
                0.72,
                rf"toy $P(t_0=0)={zero_fraction:.3f}$",
                ha="right",
                transform=ax.transAxes,
            )

    fig.tight_layout()
    return fig


def plot_mu_hat_toys(toy_results, sigma_by_mu, n_bins=35):
    """Compare prepared bounded estimators with the Wald/Cowan prediction."""
    hypotheses = sorted(toy_results["mu_true"].unique())
    fig, axes = plt.subplots(1, len(hypotheses), figsize=(6.2 * len(hypotheses), 4.6))
    axes = np.atleast_1d(axes)

    for ax, mu_true in zip(axes, hypotheses):
        values = toy_results.loc[toy_results["mu_true"] == mu_true, "mu_hat"].to_numpy()
        sigma = float(sigma_by_mu[float(mu_true)])
        upper = max(
            float(mu_true) + 5.0 * sigma,
            5.0 * sigma,
            1.001 * float(values.max()),
        )
        edges = np.linspace(0.0, upper, n_bins + 1)
        toy_probability = np.histogram(values, bins=edges)[0] / len(values)
        gaussian_cdf = norm.cdf((edges - float(mu_true)) / sigma)
        asymptotic_probability = np.diff(gaussian_cdf)
        boundary_mass = float(norm.cdf(-float(mu_true) / sigma))
        asymptotic_probability[0] += boundary_mass

        ax.stairs(
            toy_probability,
            edges,
            color="C2",
            lw=2,
            fill=True,
            alpha=0.22,
            label="flow toys",
        )
        ax.stairs(
            asymptotic_probability,
            edges,
            color="C3",
            lw=2,
            ls="--",
            label=rf"bounded Wald ($\sigma_\mu={sigma:.3g}$)",
        )
        ax.axvline(float(mu_true), color="black", ls=":", lw=1.5)
        ax.set_xlabel(r"$\hat\mu$")
        ax.set_ylabel("probability / bin")
        ax.set_title(rf"toys generated at $\mu={mu_true:g}$")
        ax.legend(fontsize=9)

    fig.tight_layout()
    return fig
