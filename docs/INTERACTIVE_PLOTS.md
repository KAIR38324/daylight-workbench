# Offline interactive plots

`InteractiveParetoWidget` loads a local Plotly.js bundle in a QtWebEngine view. The stylesheet, HTML and Plotly MIT license live in `daylight_workbench/plot_assets/`. The 4.7 MB Plotly JS file is copied from the installed Plotly wheel at build time, so it need not be stored in source control. Source runs can use an installed Plotly wheel to make a temporary local copy automatically. A packaged app needs only the copied JS, not the Python Plotly runtime. No CDN is used. The WebEngine profile blocks remote requests, navigation outside the bundled page and local files outside the asset directory. The HTML Content Security Policy blocks network connections. Keep the QtWebEngine sandbox enabled.

## Integrate

```python
from daylight_workbench.interactive_plot import InteractiveParetoWidget

plot = InteractiveParetoWidget(parent)
plot.set_data(candidates, objective_columns=['predicted_UDI', 'predicted_DGP'],
              input_columns=['window_ratio', 'shade_depth'],
              id_column='candidate_id', pareto_column='pareto')
plot.selectionChanged.connect(select_table_rows_by_candidate_id)
# When the table selection changes, including after sorting or filtering:
plot.set_selection(selected_candidate_ids_from_table)
```

`set_data` requires two or more numeric objectives and unique, non-null candidate IDs. `input_columns` is optional; external result sets with only objectives work without fabricated design variables. A categorical input gets visible category ticks in the parallel-coordinates plot. IDs retain their JSON scalar type. `selectionChanged` carries IDs, never displayed row indices. Internal opaque selection keys retain very large numeric IDs exactly through JavaScript. `set_selection` does not re-emit the signal. Replacing a dataset whose ID set or objective columns change clears the selection. A same-ID progress refresh retains selection.

The compact tabs provide the 2D/3D Pareto scatter plus parallel-coordinate brushing, an objective-pair scatter matrix (first four objective columns, labeled when more exist), a per-objective distribution with selected-solution position markers, a per-objective selected-solution comparison in original units (first eight selections are shown and labeled if more are selected), and a numeric input-versus-objective Pearson correlation heatmap. Correlations are computed on the current candidate population with pairwise complete observations; missing or constant pairs show undefined, categorical inputs are excluded, and correlation is explicitly not causal. Matrix points and histogram bins can also select candidates. 2D lasso/click and 3D click select solutions; 3D rotation is supported when WebGL is available. Without WebGL, the widget uses an actual SVG scatter and a brushable SVG parallel-coordinates view; 3D is explicitly disabled rather than showing a blank WebGL warning. Brushing a parallel axis filters the selection linked to the scatter and the caller's table. Axis dropdowns choose displayed objectives. All rows are sent to the browser, with no silent sampling; the footer states how many have nonmissing values on the selected scatter axes. Plot selection never changes the underlying table or export scope. Hover shows candidate ID and original objective values. Pareto coloring only appears when an actual `pareto` column is provided. The module deliberately does not compute hypervolume or renormalize it during progress updates. All selected-solution highlights and other charts use the same stable-ID state; a chart-only display bound never changes the table or export scope.

For a live optimizer snapshot containing `X`, `Y`, `inputs`, `targets`, `generation` and `evaluations`, the application should convert the arrays to a stable-ID DataFrame, derive true Pareto membership if desired, and call `set_data` on the GUI thread at a throttled cadence. The caller should show generation/evaluation progress outside this widget; this view visualizes the current frontier. Avoid assigning new IDs to existing candidates at each update.

## Windows packaging

First run `python -m daylight_workbench.interactive_plot --prepare-assets` in the build environment after installing the pinned Plotly package. Then, for PyInstaller, add `--add-data "daylight_workbench/plot_assets;daylight_workbench/plot_assets"` on Windows. Retain PyInstaller's PySide6 QtWebEngine hooks so `QtWebEngineProcess` and Qt WebEngine resources ship with the executable. Do not use `--disable-web-security`, `--no-sandbox` or remote Plotly scripts. Validate on a clean, disconnected Windows system: load results, brush, switch to 3D, and verify there are no remote requests. A Linux offscreen QtWebEngine smoke test exists, but it is not a substitute for that Windows packaged test.
