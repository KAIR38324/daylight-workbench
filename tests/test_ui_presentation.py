"""Presentation-only regressions. Fixtures are synthetic, with no provider calls."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pandas as pd
from PySide6 import QtCore as C, QtWidgets as W

from daylight_workbench.external_results import prepare_external_results
from daylight_workbench.modeling import LayerSpec
from daylight_workbench.ui import FrameModel, MainWindow


def application():
    return W.QApplication.instance() or W.QApplication([])


def test_sort_preserves_selection_by_candidate_id_not_row_number():
    app = application()
    window = MainWindow()
    context = prepare_external_results(pd.DataFrame({'x': [2., 3., 1.], 'y': [3., 1., 2.]}),
                                       [], ['x', 'y'], {'x': 'min', 'y': 'min'}, 'synthetic-test')
    window.external_ready(context)
    window.result_source.setCurrentIndex(1)
    window.apply_filter()
    window.results_table.selectRow(1)
    chosen = window.selected_frame().candidate_id.tolist()
    assert chosen
    window.results_table.sortByColumn(1, C.Qt.SortOrder.DescendingOrder)
    assert window.selected_frame().candidate_id.tolist() == chosen
    assert window.interactive_plot.selected_ids() == chosen
    window.results_table.sortByColumn(2, C.Qt.SortOrder.AscendingOrder)
    assert window.selected_frame().candidate_id.tolist() == chosen
    assert window.interactive_plot.selected_ids() == chosen
    pd.testing.assert_frame_equal(context.frame[['x', 'y']], pd.DataFrame({'x': [2., 3., 1.], 'y': [3., 1., 2.]}))
    window.close()
    app.processEvents()


def test_layer_cards_remain_readable_and_share_exact_specification():
    app = application()
    window = MainWindow()
    window.layer_builder.set_layers([LayerSpec(12+i, 'tanh') for i in range(8)])
    window.layer_builder.set_io(10, 3)
    assert len(window.layer_builder.rows) == 8
    assert not window.layer_builder.add_button.isEnabled()
    assert all(card.width() == 230 for card, _, _ in window.layer_builder.rows)
    assert [s.width for s in window.layer_builder.specification()] == list(range(12, 20))
    assert window.layer_builder.json_spec() == [{'width': 12+i, 'activation': 'tanh'} for i in range(8)]
    window.layer_builder.remove_layer(window.layer_builder.rows[3])
    assert window.layer_builder.add_button.isEnabled()
    assert window.layer_builder.rows[3][0].findChild(W.QLabel, 'layerTitle').text() == '隐藏层 04'
    window.close()
    app.processEvents()


def test_native_loss_summary_displays_only_snapshot_values():
    app = application()
    window = MainWindow()
    window.handle_snapshot({'kind': 'loss', 'family': 'ann', 'epoch': 2, 'best_epoch': 1,
                            'early_stopped': False, 'history': [
                                {'epoch': 1, 'train_loss': 1.3, 'validation_loss': .9},
                                {'epoch': 2, 'train_loss': 1.1, 'validation_loss': .95}]})
    assert window.training_stats['epoch'].text() == '2'
    assert window.training_stats['best'].text() == '1'
    assert window.training_stats['loss'].text() == '0.95'
    assert window.training_figure.axes[0].lines[0].get_ydata().tolist() == [1.3, 1.1]
    window.handle_snapshot({'kind': 'fit', 'family': 'rf', 'tree': 4, 'trees': 12})
    assert window.training_stats['loss'].text() == '不适用'
    assert window.training_stats['epoch'].text() == '4 / 12'
    assert not window.training_figure.axes[0].lines  # no invented RF loss curve
    window.close()
    app.processEvents()


def test_table_formatting_keeps_full_values_in_tooltips():
    application()
    frame = pd.DataFrame({'long_name': ['synthetic_objective_with_a_long_name'], 'value': [1.234567891]})
    model = FrameModel(frame)
    assert model.data(model.index(0, 1)) == '1.23457'
    assert model.data(model.index(0, 1), C.Qt.ItemDataRole.ToolTipRole) == '1.234567891'
    assert model.data(model.index(0, 0), C.Qt.ItemDataRole.ToolTipRole) == frame.iloc[0, 0]
    pd.testing.assert_frame_equal(model.frame, frame)
