"""Offline plot payload, bridge security, and QtWebEngine smoke tests."""
import os
import sys
import time
from pathlib import Path

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pandas as pd
import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from daylight_workbench.interactive_plot import (
    ASSET_DIR, InteractiveParetoWidget, _LocalOnlyInterceptor,
    prepare_plot_payload,
)


def test_payload_keeps_every_id_and_escapes_hover():
    frame = pd.DataFrame({
        'candidate_id': ['<img src=x onerror=alert(1)>', 'b', 'c'],
        'glare <script>': [0.2, 0.3, float('nan')],
        'daylight': [60, 55, 58],
        'orientation': ['east', 'west', 'east'],
        'pareto': ['true', 'false', 'true'],
    })
    original = frame.copy(deep=True)
    data = prepare_plot_payload(frame, ['glare <script>', 'daylight'], ['orientation'])
    assert data['count'] == 3 and len(data['rows']) == 3
    assert data['rows'][0]['hover_id'] == '&lt;img src=x onerror=alert(1)&gt;'
    assert data['rows'][0]['id'] == frame.iloc[0]['candidate_id']
    assert data['rows'][0]['pareto'] is True and data['rows'][1]['pareto'] is False
    assert data['rows'][2]['values']['glare <script>'] is None
    assert data['dimensions'][-1]['categories'] == ['east', 'west']
    pd.testing.assert_frame_equal(frame, original)


def test_payload_rejects_ambiguous_ids_and_invalid_columns():
    base = pd.DataFrame({'candidate_id': ['a', 'a'], 'x': [1, 2], 'y': [3, 4]})
    with pytest.raises(ValueError, match='unique'):
        prepare_plot_payload(base, ['x', 'y'])
    with pytest.raises(ValueError, match='Missing columns'):
        prepare_plot_payload(base.iloc[:1], ['x', 'absent'])
    with pytest.raises(ValueError, match='At least two'):
        prepare_plot_payload(base.iloc[:1], ['x'])


def test_plot_assets_and_remote_request_blocked():
    index = (ASSET_DIR / 'index.html').read_text(encoding='utf-8')
    assert (ASSET_DIR / 'index.html').is_file()
    assert 'plotly.min.js' in index and 'qrc:///qtwebchannel/qwebchannel.js' in index
    assert 'connect-src \'none\'' in index
    assert not any(token in index for token in ('cdn.', 'http://', 'https://'))
    class Info:
        def __init__(self, url): self.url, self.blocked = QUrl(url), False
        def requestUrl(self): return self.url
        def block(self, value): self.blocked = value
    interceptor = _LocalOnlyInterceptor()
    for url in ['https://plotly.com/x', 'http://localhost:8000/x', 'file:///tmp/secret.txt']:
        info = Info(url)
        interceptor.interceptRequest(info)
        assert info.blocked
    info = Info(QUrl.fromLocalFile(str(ASSET_DIR / 'plotly.min.js')).toString())
    interceptor.interceptRequest(info)
    assert not info.blocked


@pytest.fixture(scope='module')
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _wait(app, predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.025)
    return False


def _js(app, widget, expression):
    results = []
    widget.view.page().runJavaScript(expression, results.append)
    assert _wait(app, lambda: bool(results)), f'JavaScript did not finish: {expression}'
    return results[0]


def test_qtwebengine_renders_and_roundtrips_stable_ids(app):
    # This tests a real local-file QtWebEngine load, not just the HTML template.
    widget = InteractiveParetoWidget()
    widget.resize(1000, 700)
    widget.show()
    frame = pd.DataFrame({'candidate_id': ['x<1>', 'b', 'c'],
                          'obj one': [1., 2., 3.], 'obj two': [4., 3., 2.],
                          'pareto': [True, False, True]})
    widget.set_data(frame, ['obj one', 'obj two'])
    assert _wait(app, lambda: widget._bridge.ready_for_data)
    assert _wait(app, lambda: _js(app, widget, "document.getElementById('total').textContent") == '3')
    assert _wait(app, lambda: _js(app, widget, "document.querySelectorAll('.js-plotly-plot').length") >= 1)
    assert _js(app, widget, "document.querySelector('#scatter').data.length") == 3
    svg_fallback = _js(app, widget, "document.querySelector('#parallel svg') !== null") is True
    if not svg_fallback:
        assert _js(app, widget, "document.querySelector('#parallel').data.length") == 1
    received = []
    widget.selectionChanged.connect(received.append)
    _js(app, widget, "document.querySelector('#scatter').emit('plotly_click',{points:[{customdata:document.querySelector('#scatter').data[0].customdata[0]}]})")
    assert _wait(app, lambda: received == [['b']])
    assert widget.selected_ids() == ['b']
    # The no-WebGL SVG renderer remains a genuine axis-brush selection surface.
    if svg_fallback:
        _js(app, widget, "(() => {const h=document.querySelector('[data-brush-dimension=\"obj one\"]');const r=h.getBoundingClientRect();const y=f=>r.top+r.height*f;h.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,clientY:y(-.02),pointerId:17}));h.dispatchEvent(new PointerEvent('pointerup',{bubbles:true,clientY:y(.55),pointerId:17}));return 'brushed';})()")
        assert _wait(app, lambda: widget.selected_ids() == ['b', 'c'])
    widget.set_selection(['x<1>'])
    assert _wait(app, lambda: _js(app, widget, "document.getElementById('selected').textContent") == '1')
    widget._receive_selection('{"version":1,"keys":["unknown"]}')
    assert widget.selected_ids() == ['x<1>']
    widget.set_data(frame.iloc[:2], ['obj one', 'obj two'])
    assert widget.selected_ids() == []  # Changed dataset clears stale selection.
    widget.close()


def test_qtwebengine_opaque_keys_preserve_numeric_ids(app):
    widget = InteractiveParetoWidget()
    widget.resize(800, 600)
    widget.show()
    large = 2**60
    frame = pd.DataFrame({'candidate_id': [large, large + 1],
                          'x': [1., 2.], 'y': [4., 3.]})
    widget.set_data(frame, ['x', 'y'])
    assert _wait(app, lambda: widget._bridge.ready_for_data)
    assert _wait(app, lambda: _js(app, widget, "document.querySelector('#scatter')?.data?.[0]?.customdata?.length") == 2)
    _js(app, widget, "document.querySelector('#scatter').emit('plotly_click',{points:[{customdata:document.querySelector('#scatter').data[0].customdata[1]}]})")
    assert _wait(app, lambda: widget.selected_ids() == [large + 1])
    widget.close()


def test_correlation_population_not_causation_and_no_fake_inputs():
    f = pd.DataFrame({'candidate_id': ['a', 'b', 'c'],
                      'UDI': [1., 2., 3.], 'DGP': [3., 2., 1.],
                      'window': [2., 4., 6.], 'facade': ['N', 'S', 'E']})
    data = prepare_plot_payload(f, ['UDI', 'DGP'], ['window', 'facade'])
    assert data['correlations'][0]['input'] == 'window'
    assert data['correlations'][0]['cells'][0]['r'] == pytest.approx(1)
    assert data['correlations'][0]['cells'][1]['r'] == pytest.approx(-1)
    assert data['excluded_inputs'] == ['facade']
    objective_only = prepare_plot_payload(f, ['UDI', 'DGP'])
    assert objective_only['inputs'] == [] and objective_only['correlations'] == []


def test_internal_tabs_and_table_selection_highlight(app):
    widget = InteractiveParetoWidget()
    widget.resize(1000, 680)
    widget.show()
    frame = pd.DataFrame({'candidate_id': ['a', 'b', 'c', 'd'],
                          'UDI': [1., 2., 3., 4.], 'DGP': [4., 3., 2., 1.],
                          'Energy': [2., 4., 3., 1.], 'WWR': [.1, .2, .3, .4],
                          'pareto': [True, False, True, True]})
    widget.set_data(frame, ['UDI', 'DGP', 'Energy'], ['WWR'])
    assert _wait(app, lambda: widget._bridge.ready_for_data)
    assert _wait(app, lambda: _js(app, widget, "document.querySelector('#scatter')?.data?.length") == 3)
    widget.set_selection(['b'])  # As if from a sorted/filtered results table.
    assert _wait(app, lambda: _js(app, widget, "document.querySelector('#scatter').data[2].x.length") == 1)
    assert _js(app, widget, "document.querySelector('#scatter').data[0].marker.opacity") == pytest.approx(.17)
    _js(app, widget, "document.getElementById('mode2').click()")
    assert _wait(app, lambda: _js(app, widget, "document.querySelector('#scatter').data[2].x.length") == 1)
    if _js(app, widget, "!document.getElementById('mode3').disabled"):
        _js(app, widget, "document.getElementById('mode3').click()")
        assert _wait(app, lambda: _js(app, widget, "document.querySelector('#scatter').data[2].type") == 'scatter3d')
        assert _js(app, widget, "document.querySelector('#scatter').data[2].x.length") == 1
        _js(app, widget, "document.getElementById('mode2').click()")
        assert _wait(app, lambda: _js(app, widget, "document.querySelector('#scatter').data[2].type") in ('scatter', 'scattergl'))
    else:
        assert '3D 不可用' in _js(app, widget, "document.getElementById('scatterHint').textContent")
    for view, chart, minimum in [('matrix', 'matrixChart', 1),
                                  ('distribution', 'distributionChart', 1),
                                  ('comparison', 'comparisonChart', 1),
                                  ('correlation', 'correlationChart', 1)]:
        _js(app, widget, f"document.querySelector('[data-analysis-view=\"{view}\"]').click()")
        assert _wait(app, lambda c=chart: _js(app, widget, f"document.querySelector('#{c} svg') !== null") is True)
        assert _js(app, widget, f"document.querySelectorAll('#{chart} svg').length") >= minimum
    assert '不代表因果关系' in _js(app, widget, "document.getElementById('correlationNote').textContent")
    _js(app, widget, "document.querySelector('[data-analysis-view=\"matrix\"]').click()")
    assert _wait(app, lambda: _js(app, widget, "document.querySelectorAll('#matrixChart circle[fill=\"#f2a430\"]').length") > 0)
    _js(app, widget, "Array.from(document.querySelectorAll('#matrixChart circle')).find(e=>e.getAttribute('data-candidate-key')===JSON.stringify('c')).dispatchEvent(new MouseEvent('click',{bubbles:true}))")
    assert _wait(app, lambda: widget.selected_ids() == ['b', 'c'])
    _js(app, widget, "document.querySelector('[data-analysis-view=\"distribution\"]').click()")
    assert _wait(app, lambda: _js(app, widget, "document.querySelectorAll('#distributionChart line[stroke=\"#f2a430\"]').length") >= 2)
    _js(app, widget, "Array.from(document.querySelectorAll('#distributionChart rect[data-hist-bin]')).find(e=>Number(e.getAttribute('data-count'))>0).dispatchEvent(new MouseEvent('click',{bubbles:true}))")
    assert _wait(app, lambda: len(widget.selected_ids()) > 0)
    # Switching tabs does not discard the browser-originated selected IDs.
    _js(app, widget, "document.querySelector('[data-analysis-view=\"frontier\"]').click()")
    assert _wait(app, lambda: _js(app, widget, "document.querySelector('#scatter').data[2].x.length") == 1)
    assert widget.selected_ids()
    widget.close()


def test_webgl_parallel_trace_restyle_contract_without_gpu():
    """Exercise JS trace construction/call with a fake Plotly, not real WebGL."""
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is unavailable for the JavaScript unit harness')
    harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map(),restyles=[];
function element(id) {if(!elements.has(id))elements.set(id,{
 id,style:{},children:[],handlers:{},clientWidth:900,clientHeight:500,
 classList:{toggle(){}},replaceChildren(){this.children=[]},appendChild(v){this.children.push(v)},
 addEventListener(){},on(name,fn){this.handlers[name]=fn},setAttribute(){},
});return elements.get(id)}
const bridge={payloadChanged:{connect(fn){this.cb=fn}},selectionChanged:{connect(fn){this.cb=fn}},ready(){},reportSelection(raw){throw Error('unexpected selection report: '+raw)}};
const context={
 document:{getElementById:element,querySelectorAll(){return []},querySelector(){return element('button')},
 createElement(tag){if(tag==='canvas')return {getContext(){return {}}};return element('new-'+tag)},
 createTextNode(text){return {text}}},
 window:{addEventListener(){}},qt:{webChannelTransport:{}},
 QWebChannel:function(_transport,fn){fn({objects:{plotBridge:bridge}})},
 Plotly:{react(el,traces){el.data=traces;return Promise.resolve()},restyle(el,update){
 restyles.push(update);el.data[0].line.color=update['line.color'][0];return Promise.resolve()}},
};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context,{filename:'plot.js'});
const rows=[
 {id:'a',id_key:'"a"',display_id:'a',hover_id:'a',pareto:true,values:{x:1,y:3}},
 {id:'b',id_key:'"b"',display_id:'b',hover_id:'b',pareto:false,values:{x:2,y:2}},
];
bridge.payloadChanged.cb(JSON.stringify({rows,count:2,version:1,objectives:['x','y'],inputs:[],
 dimensions:[{name:'x',kind:'numeric'},{name:'y',kind:'numeric'}],has_pareto:true}));
(async()=>{await new Promise(setImmediate);
 const plot=element('parallel');
 assert.equal(plot.data[0].type,'parcoords');
 assert.equal(plot.data[0].line.color.join(','),'1,0');
 assert.equal(plot.data[0].line.colorscale.at(-1)[1],'#f2a430');
 restyles.length=0;
 bridge.selectionChanged.cb(JSON.stringify({version:1,keys:['"b"']}));
 assert.equal(plot.data[0].line.color.join(','),'1,2');
 assert.ok(restyles.some(v=>v['line.color'][0][1]===2));
 // A line-color restyle must not be interpreted as a new axis brush.
 plot.handlers.plotly_restyle([{'line.color':[[1,2]]},[0]]);
 assert.equal(restyles.length,1);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run([node, '-e', harness, str(ASSET_DIR / 'plot.js')],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
