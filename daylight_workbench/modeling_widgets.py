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
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)
        top = W.QHBoxLayout()
        title = W.QLabel('网络结构')
        title.setProperty('role', 'sectionTitle')
        title.setToolTip('逐层编辑；图示与模型训练使用同一结构规格。')
        top.addWidget(title)
        self.structure_summary = W.QLabel()
        self.structure_summary.setProperty('role', 'muted')
        top.addWidget(self.structure_summary)
        top.addStretch()
        self.add_button = W.QPushButton('+ 添加隐藏层')
        self.add_button.clicked.connect(lambda: self.add_layer())
        top.addWidget(self.add_button)
        root.addLayout(top)
        cards_host = W.QWidget()
        self.cards = W.QHBoxLayout(cards_host)
        self.cards.setContentsMargins(0, 0, 0, 0)
        self.cards.setSpacing(10)
        self.cards.setAlignment(C.Qt.AlignmentFlag.AlignLeft)
        self.card_scroll = W.QScrollArea()
        self.card_scroll.setWidgetResizable(True)
        self.card_scroll.setWidget(cards_host)
        self.card_scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        self.card_scroll.setFixedHeight(166)
        self.card_scroll.setVerticalScrollBarPolicy(C.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(self.card_scroll)
        self.rows = []
        self.scene = W.QGraphicsScene(self)
        self.diagram = W.QGraphicsView(self.scene)
        self.diagram.setRenderHints(G.QPainter.RenderHint.Antialiasing | G.QPainter.RenderHint.TextAntialiasing)
        self.diagram.setMinimumHeight(150)
        self.diagram.setMaximumHeight(180)
        self.diagram.setHorizontalScrollBarPolicy(C.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.diagram.setVerticalScrollBarPolicy(C.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.diagram.setToolTip('结构预览与模型使用同一份层规格；箭头仅表示顺序连接，不可自由拖线。')
        root.addWidget(self.diagram)
        self.io = (0, 0)
        self._loading = False
        self.set_layers([LayerSpec(64), LayerSpec(64)])

    def add_layer(self, layer=None):
        if len(self.rows) >= 8:
            return
        layer = layer if isinstance(layer, LayerSpec) else LayerSpec()
        card = W.QFrame()
        card.setProperty('role', 'card')
        card.setFixedWidth(230)
        card.setFixedHeight(150)
        layout = W.QVBoxLayout(card)
        layout.setContentsMargins(13, 10, 13, 12)
        layout.setSpacing(7)
        heading = W.QHBoxLayout()
        title = W.QLabel()
        title.setObjectName('layerTitle')
        title.setProperty('role', 'eyebrow')
        heading.addWidget(title)
        heading.addStretch()
        width = W.QSpinBox()
        width.setRange(1, 2048)
        width.setValue(layer.width)
        activation = W.QComboBox()
        activation.addItems(ACTIVATIONS)
        activation.setCurrentText(layer.activation)
        remove = W.QPushButton('移除')
        remove.setObjectName('removeLayer')
        remove.setProperty('role', 'quiet')
        remove.setToolTip('移除此隐藏层，至少保留一层。')
        heading.addWidget(remove)
        layout.addLayout(heading)
        form = W.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.addRow('神经元', width)
        form.addRow('激活函数', activation)
        layout.addLayout(form)
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_diagram()

    def _fit_diagram(self):
        if not hasattr(self, 'scene') or not self.scene.items():
            return
        rect = self.scene.sceneRect()
        # Never enlarge a small network until text becomes poster-sized. Long
        # networks retain a readable minimum scale and can scroll horizontally.
        scale = min(1.0, max(.72, (self.diagram.viewport().width() - 16) / max(1, rect.width())))
        self.diagram.setTransform(G.QTransform.fromScale(scale, scale))

    def draw_diagram(self):
        self.scene.clear()
        layers = [{'width': self.io[0], 'activation': 'Input', 'title': '输入层'}]
        layers += [dict(layer, title=f'隐藏层 {i+1:02d}') for i, layer in enumerate(self.json_spec())]
        layers += [{'width': self.io[1], 'activation': 'Linear', 'title': '输出层'}]
        for i, (card, _, _) in enumerate(self.rows):
            card.findChild(W.QLabel, 'layerTitle').setText(f'隐藏层 {i+1:02d}')
            card.findChild(W.QPushButton, 'removeLayer').setEnabled(len(self.rows) > 1)
        self.add_button.setEnabled(len(self.rows) < 8)
        sizes = [layer['width'] for layer in layers]
        params = sum((a + 1) * b for a, b in zip(sizes, sizes[1:])) if all(sizes) else None
        self.structure_summary.setText(f'{len(self.rows)} 个隐藏层' + (f' · {params:,} 个参数' if params is not None else ' · 待定义输入与目标'))
        small = G.QFont(self.font())
        small.setPointSizeF(8.5)
        number = G.QFont(self.font())
        number.setPointSizeF(17)
        number.setWeight(G.QFont.Weight.DemiBold)
        for i, layer in enumerate(layers):
            x = i * 166
            path = G.QPainterPath()
            path.addRoundedRect(C.QRectF(x, 0, 132, 98), 6, 6)
            self.scene.addPath(path, G.QPen(G.QColor('#d2dee8')), G.QBrush(G.QColor('#ffffff')))
            for value, font, color, yy in [(layer['title'], small, '#6d7f90', 10),
                                          (str(layer['width']) if layer['width'] else '—', number, '#24475f', 31),
                                          (layer['activation'], small, '#6d7f90', 70)]:
                text = self.scene.addSimpleText(value, font)
                text.setBrush(G.QBrush(G.QColor(color)))
                text.setPos(x + 14, yy)
            if i:
                pen = G.QPen(G.QColor('#9cafbe'), 1.4)
                self.scene.addLine(x - 30, 49, x - 7, 49, pen)
                self.scene.addLine(x - 12, 45, x - 7, 49, pen)
                self.scene.addLine(x - 12, 53, x - 7, 49, pen)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-18, -16, 18, 16))
        self._fit_diagram()


class SearchControls(W.QWidget):
    changed = C.Signal()

    def __init__(self, builder, parent=None):
        super().__init__(parent)
        self.builder = builder
        root = W.QGridLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setHorizontalSpacing(18)
        root.setVerticalSpacing(12)
        self.family = W.QComboBox()
        for name, value in [('ANN 神经网络', 'ann'), ('RF 随机森林', 'rf'), ('XGBoost 梯度提升', 'xgboost')]:
            self.family.addItem(name, value)
        self.compare_all = W.QCheckBox('比较 ANN、RF 与 XGBoost')
        self.compare_all.setToolTip('三类模型共用同一折清单，分别使用右侧明确配置的网格。')
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
        def group(title, items):
            box = W.QGroupBox(title)
            form = W.QFormLayout(box)
            form.setVerticalSpacing(10)
            form.setHorizontalSpacing(16)
            for text, widget in items:
                form.addRow(text, widget)
            return box
        validation = group('模型与验证', [('模型', self.family), ('跨模型比较', self.compare_all),
                            ('验证方案', self.mode), ('CV 折数', self.fold_count),
                            ('实际拟合预算上限', self.max_fits), ('搜索种子', self.seed), ('CPU 线程上限', self.threads)])
        self.max_fits.setToolTip('预算包含 XGBoost 逐目标拟合与最终重拟合；超预算会拒绝运行。')
        root.addWidget(validation, 0, 0, 2, 1)
        self.model_groups = {
            'ann': group('ANN · 整体架构候选', [('架构模板（可选）', self.templates), ('学习率', self.ann_rates), ('批量大小', self.ann_batch)]),
            'rf': group('RF · 随机森林', [('树数', self.rf_trees), ('最大深度', self.rf_depth)]),
            'xgboost': group('XGBoost · 梯度提升', [('迭代数', self.xgb_rounds), ('最大深度', self.xgb_depth), ('学习率', self.xgb_rates)]),
        }
        root.addWidget(self.model_groups['ann'], 0, 1)
        root.addWidget(self.model_groups['rf'], 1, 1)
        root.addWidget(self.model_groups['xgboost'], 2, 1)
        note = W.QLabel('候选值用逗号分隔。运行前先预览组合数与拟合预算；测试集不参与选优。')
        note.setWordWrap(True)
        note.setProperty('role', 'muted')
        root.addWidget(note, 2, 0)
        root.setRowStretch(3, 1)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 1)
        self.family.currentIndexChanged.connect(self._refresh_model_groups)
        self.mode.currentIndexChanged.connect(self._refresh_model_groups)
        self.compare_all.toggled.connect(self._refresh_model_groups)
        self._refresh_model_groups()
        for widget in [self.family, self.compare_all, self.mode, self.fold_count, self.max_fits,
                       self.templates, self.ann_rates, self.ann_batch, self.rf_trees, self.rf_depth,
                       self.xgb_rounds, self.xgb_depth, self.xgb_rates, self.seed, self.threads]:
            if isinstance(widget, W.QLineEdit):
                widget.textChanged.connect(self.changed)
            elif isinstance(widget, W.QComboBox):
                widget.currentIndexChanged.connect(self.changed)
            elif isinstance(widget, W.QCheckBox):
                widget.toggled.connect(self.changed)
            else:
                widget.valueChanged.connect(self.changed)

    def _refresh_model_groups(self, *args):
        for family, group in self.model_groups.items():
            group.setEnabled(self.compare_all.isChecked() or self.family.currentData() == family)
        self.fold_count.setEnabled(self.mode.currentData() == 'cv')

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
