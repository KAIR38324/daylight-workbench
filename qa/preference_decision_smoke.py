"""Synthetic preference-page screenshots, exact outbound preview, no API call.

QT_QPA_PLATFORM=offscreen ... .venv/bin/python -m qa.preference_decision_smoke
No claim of interactive GPU 3D render verification is made by this smoke.
"""
import json
from pathlib import Path

import pandas as pd
from PySide6 import QtCore as C, QtWidgets as W
from daylight_workbench.ui import MainWindow
from daylight_workbench.theme import apply_theme
from daylight_workbench.external_results import prepare_external_results
from daylight_workbench.preference_engine import default_preference_plan


def main():
    app = W.QApplication.instance() or W.QApplication([])
    apply_theme(app)
    out = Path(__file__).parent / "evidence" / "preference-decision"
    out.mkdir(parents=True, exist_ok=True)
    window = MainWindow()
    window.show()
    raw = pd.DataFrame({"UDI": [90., 70., 40., 55., 65.], "EUI": [20., 44., 100., 70., 55.], "DGP": [.8, .38, .1, .7, .5]})
    context = prepare_external_results(raw, [], list(raw.columns), {"UDI": "max", "EUI": "min", "DGP": "min"}, "SYNTHETIC UI ONLY")
    window.external_ready(context)
    p = window.decision_page
    p.confirm_mapping()
    window.tabs.setCurrentWidget(p)
    plan = default_preference_plan(p._objectives)
    plan.update(intent="能耗与眩光等权折中，采光保留作对照", explanation="按已确认方向归一化后，能耗与眩光各占 50%；采光不参与当前打分。权重可改，不存在唯一的物理最佳。", assumptions=["等权是待审核的偏好假设，不是物理结论。", "完整候选固定 min–max 归一化；保留原始 Pareto 标记。"])
    plan["weights"] = [{"column": o.column, "weight": v} for o, v in zip(p._objectives, [0, .5, .5])]
    p._load_plan_editor(plan)
    p.chat.appendPlainText("你：我要能耗和眩光平衡的方案\n\n界面测试（模拟建议，未联网）：已确认 EUI 表示能耗、DGP 表示眩光，均为最小化。建议各 50%，其余 0，可在右侧修改。")

    def capture(name):
        for _ in range(8):
            app.processEvents()
        assert window.grab().save(str(out / name))

    capture("review-1440.png")
    p.apply_reviewed_plan()
    capture("decision-1440.png")
    window.resize(1100, 760)
    capture("decision-1100.png")
    p.detail_tabs.setCurrentIndex(0)
    capture("review-1100.png")
    p.query_edit.setPlainText("帮我选更考虑采光的")
    p.make_preview()
    capture("exact-preview.png")
    evidence = {"synthetic": True, "live_api": False, "tab_count": window.tabs.count(),
                "ranking_ids": p.ranking.frame.candidate_id.tolist(), "scores": p.ranking.metrics.preference_score.tolist(),
                "selected_ids": p.selected_ids(), "bounds": p.ranking.bounds,
                "offscreen_3d_visual_verified": False}
    (out / "smoke.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))
    window.close()
    window.deleteLater()
    C.QCoreApplication.sendPostedEvents(None, C.QEvent.Type.DeferredDelete)
    app.processEvents()


if __name__ == "__main__":
    main()
