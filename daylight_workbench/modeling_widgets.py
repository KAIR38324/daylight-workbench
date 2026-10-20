"""Native Qt layer cards, same-spec network diagram, and bounded grid inputs."""
from dataclasses import asdict
from itertools import product
import json
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W
from .modeling import ACTIVATIONS, LayerSpec, ModelSpec, SearchConfig


class ANNLayerBuilder(W.QWidget):
    changed = C.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = W.QVBoxLayout(self)
        top = W.QHBoxLayout()
        title = W.QLabel('ANN 隐藏层')
        title.setToolTip('逐层编辑；图示与模型训练使用同一结构规格。')
        top.addWidget(title)
        add = W.QPushButton('+ 添加隐藏层')
        add.clicked.connect(lambda: self.add_layer())
        top.addWidget(add)
        root.addLayout(top)
        self.cards = W.QHBoxLayout()
        root.addLayout(self.cards)
        self.rows = []
        self.scene = W.QGraphicsScene(self)
        self.diagram = W.QGraphicsView(self.scene)
        self.diagram.setMinimumHeight(110)
        self.diagram.setMaximumHeight(140)
        root.addWidget(self.diagram)
        self.io = (0, 0)
        self._loading = False
        self.set_layers([LayerSpec(64), LayerSpec(64)])

    def add_layer(self, layer=None):
        if len(self.rows) >= 8:
            return
        layer = layer if isinstance(layer, LayerSpec) else LayerSpec()
        card = W.QFrame()
        card.setFrameShape(W.QFrame.Shape.StyledPanel)
        layout = W.QVBoxLayout(card)
        width = W.QSpinBox()
        width.setRange(1, 2048)
        width.setValue(layer.width)
        activation = W.QComboBox()
        activation.addItems(ACTIVATIONS)
        activation.setCurrentText(layer.activation)
        remove = W.QPushButton('删除此层')
        layout.addWidget(W.QLabel('神经元数'))
        layout.addWidget(width)
        layout.addWidget(activation)
        layout.addWidget(remove)
        row = (card, width, activation)
        self.rows.append(row)
        self.cards.addWidget(card)
        width.valueChanged.connect(self._change)
        activation.currentIndexChanged.connect(self._change)
        remove.clicked.connect(lambda: self.remove_layer(row))
        self._change()

    def remove_layer(self, row):
        if len(self.rows) <= 1:
            return
        self.rows.remove(row)
        self.cards.removeWidget(row[0])
        row[0].hide()
        row[0].deleteLater()
        self._change()

    def set_layers(self, layers):
        self._loading = True
        for row in self.rows:
            self.cards.removeWidget(row[0])
            row[0].hide()
            row[0].deleteLater()
        self.rows = []
        for layer in layers:
            self.add_layer(layer)
        self._loading = False
        self._change()

    def specification(self):
        return [LayerSpec(width.value(), activation.currentText()) for _, width, activation in self.rows]

    def json_spec(self):
        return [asdict(layer) for layer in self.specification()]

    def set_io(self, inputs, outputs):
        self.io = (inputs, outputs)
        self.draw_diagram()

    def _change(self, *args):
        if self._loading:
            return
        self.draw_diagram()
        self.changed.emit()

    def draw_diagram(self):
        self.scene.clear()
        layers = [{'width': self.io[0], 'activation': 'input'}] + self.json_spec() + [{'width': self.io[1], 'activation': 'linear output'}]
        for i, layer in enumerate(layers):
            x = i * 125
            self.scene.addRect(x, 8, 105, 62, G.QPen(G.QColor('#52738e')), G.QBrush(G.QColor('#eaf3fa')))
            text = self.scene.addText(f"{layer['width']}\n{layer['activation']}")
            text.setPos(x + 9, 14)
            if i:
                self.scene.addLine(x - 20, 39, x, 39, G.QPen(G.QColor('#52738e'), 2))
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-4, -4, 4, 4))
        self.diagram.fitInView(self.scene.sceneRect(), C.Qt.AspectRatioMode.KeepAspectRatio)


class SearchControls(W.QWidget):
    changed = C.Signal()

    def __init__(self, builder, parent=None):
        super().__init__(parent)
        self.builder = builder
        form = W.QFormLayout(self)
        self.family = W.QComboBox()
        for name, value in [('ANN 神经网络', 'ann'), ('RF 随机森林', 'rf'), ('XGBoost 梯度提升', 'xgboost')]:
            self.family.addItem(name, value)
        self.compare_all = W.QCheckBox('比较三类模型（同一折清单；各使用下面的明确网格）')
        self.mode = W.QComboBox()
        self.mode.addItem('严格：锁定测试 + 开发集5折CV（默认）', 'cv')
        self.mode.addItem('快速：固定训练/验证/测试划分', 'fixed')
        self.fold_count = W.QSpinBox()
        self.fold_count.setRange(2, 10)
        self.fold_count.setValue(5)
        self.max_fits = W.QSpinBox()
        self.max_fits.setRange(1, 2000)
        self.max_fits.setValue(200)
        self.templates = W.QLineEdit('')
        self.templates.setPlaceholderText('空白=上方逐层规格；或整体模板64-64;128-64，不做逐层笛卡尔积')
        self.ann_rates = W.QLineEdit('0.001')
        self.ann_batch = W.QLineEdit('64')
        self.rf_trees = W.QLineEdit('100')
        self.rf_depth = W.QLineEdit('12')
        self.xgb_rounds = W.QLineEdit('150')
        self.xgb_depth = W.QLineEdit('4')
        self.xgb_rates = W.QLineEdit('0.05')
        self.seed = W.QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(42)
        self.threads = W.QSpinBox()
        self.threads.setRange(1, 4)
        self.threads.setValue(2)
        for text, widget in [('模型', self.family), ('跨模型比较', self.compare_all), ('验证方案', self.mode),
                             ('CV折数', self.fold_count), ('最大实际拟合预算（含XGB逐目标/最终重拟合）', self.max_fits),
                             ('ANN整体架构模板（可选）', self.templates), ('ANN学习率候选（逗号分隔）', self.ann_rates),
                             ('ANN批量候选', self.ann_batch), ('RF树数候选', self.rf_trees), ('RF最大深度候选', self.rf_depth),
                             ('XGB迭代数候选', self.xgb_rounds), ('XGB深度候选', self.xgb_depth), ('XGB学习率候选', self.xgb_rates),
                             ('搜索种子', self.seed), ('CPU线程上限', self.threads)]:
            form.addRow(text, widget)
            if isinstance(widget, W.QLineEdit):
                widget.textChanged.connect(self.changed)
            elif isinstance(widget, W.QComboBox):
                widget.currentIndexChanged.connect(self.changed)
            elif isinstance(widget, W.QCheckBox):
                widget.toggled.connect(self.changed)
            else:
                widget.valueChanged.connect(self.changed)

    def _list(self, field, convert):
        result = [convert(v.strip()) for v in field.text().split(',') if v.strip()]
        if not result or len(result) > 20:
            raise ValueError('每项网格需1-20个明确候选值。')
        return list(dict.fromkeys(result))

    def configuration(self, epochs, patience):
        families = ['ann', 'rf', 'xgboost'] if self.compare_all.isChecked() else [self.family.currentData()]
        candidates = []
        if 'ann' in families:
            templates = [self.builder.specification()]
            if self.templates.text().strip():
                templates = []
                for template in self.templates.text().split(';'):
                    widths = [int(v.strip()) for v in template.split('-') if v.strip()]
                    # Whole architecture templates explicitly use ReLU; the live card
                    # architecture retains each individually selected activation.
                    templates.append([LayerSpec(width, 'relu') for width in widths])
            for layers, rate, batch in product(templates, self._list(self.ann_rates, float), self._list(self.ann_batch, int)):
                candidates.append(ModelSpec('ann', {'epochs': epochs, 'patience': patience, 'learning_rate': rate, 'batch_size': batch}, layers))
        if 'rf' in families:
            for trees, depth in product(self._list(self.rf_trees, int), self._list(self.rf_depth, int)):
                candidates.append(ModelSpec('rf', {'n_estimators': trees, 'max_depth': depth}, []))
        if 'xgboost' in families:
            for rounds, depth, rate in product(self._list(self.xgb_rounds, int), self._list(self.xgb_depth, int), self._list(self.xgb_rates, float)):
                candidates.append(ModelSpec('xgboost', {'n_estimators': rounds, 'max_depth': depth, 'learning_rate': rate, 'patience': patience}, []))
        if len(candidates) > 100:
            raise ValueError(f'当前网格{len(candidates)}个候选，超过100上限。请缩小网格。')
        return SearchConfig(candidates, self.mode.currentData(), self.fold_count.value(), self.max_fits.value(), self.seed.value(), self.threads.value())
