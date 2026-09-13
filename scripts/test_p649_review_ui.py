#!/usr/bin/env python3
"""Headless UI test for p649 review.html: click, filter, persist, reload."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8778/p649_v2a_review/review.html?disk=test"
REVIEW = Path("/root/projects/hrfont/data/p649_v2a_review")
DEC_JSON = REVIEW / "decisions.test.json"
DEC_JSONL = REVIEW / "decisions.test.jsonl"


def backup() -> tuple[str | None, str | None]:
    a = DEC_JSON.read_text(encoding="utf-8") if DEC_JSON.exists() else None
    b = DEC_JSONL.read_text(encoding="utf-8") if DEC_JSONL.exists() else None
    return a, b


def restore(a: str | None, b: str | None) -> None:
    if a is not None:
        DEC_JSON.write_text(a, encoding="utf-8")
    if b is not None:
        DEC_JSONL.write_text(b, encoding="utf-8")


def empty_disk() -> None:
    payload = {
        "dataset": "fontdiffuser-p649-t295-s338-cn2west-v2a-r0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "n": 0,
        "last_stem": "",
        "decisions": {},
    }
    DEC_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    DEC_JSONL.write_text("", encoding="utf-8")


def wait_disk_n(n: int, timeout: float = 8.0) -> dict:
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        last = json.loads(DEC_JSON.read_text(encoding="utf-8"))
        if last.get("n") == n and len(last.get("decisions") or {}) == n:
            return last
        time.sleep(0.15)
    raise AssertionError(f"disk n!={n}: {last.get('n')} keys={list((last.get('decisions') or {}).keys())}")


def progress(page) -> str:
    return page.locator("#progressText").inner_text()


def list_stems(page) -> list[str]:
    return page.locator("#fontList li[data-stem]").evaluate_all(
        "els => els.map(e => e.dataset.stem)"
    )


def sign_in(page, name: str) -> None:
    page.fill("#reviewerNew", name)
    page.locator("#btnUseReviewer").click()
    page.wait_for_timeout(400)
    text = page.locator("#whoAmI").inner_text()
    if name not in text:
        raise AssertionError(f"sign-in failed: {text}")


def main() -> None:
    bak = backup()
    fails: list[str] = []
    try:
        empty_disk()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(accept_downloads=True)
            page = ctx.new_page()
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("#fontList li[data-stem]")

            # 1 load
            stems = list_stems(page)
            if len(stems) != 649:
                fails.append(f"load list {len(stems)} != 649")
            pt = progress(page)
            if "已审 0 / 649" not in pt:
                fails.append(f"progress start: {pt}")
            if "未完成" not in page.locator("#doneBadge").inner_text():
                fails.append("done badge not 未完成")
            if page.locator(".script-block h3").count() < 7:
                fails.append("script groups missing")
            labels = page.locator(".script-block h3").all_inner_texts()
            for need in ("数字", "拉丁", "扩展拉丁", "平假名", "片假名", "注音"):
                if not any(need in x for x in labels):
                    fails.append(f"missing script label {need}")
            if "尚未选择" not in page.locator("#whoAmI").inner_text():
                fails.append("identity gate missing")

            drop_stem = stems[0]
            page.locator(f"#fontList li[data-stem='{drop_stem}']").click()
            page.once("dialog", lambda dlg: dlg.accept())
            page.locator('input[name="dec"][value="drop"]').click()
            page.wait_for_timeout(500)
            disk0 = json.loads(DEC_JSON.read_text(encoding="utf-8"))
            if disk0.get("n"):
                fails.append(f"wrote without reviewer n={disk0.get('n')}")
            if "已审 0 / 649" not in progress(page):
                fails.append(f"progress after blocked click: {progress(page)}")

            sign_in(page, "alice")
            if "alice" not in page.locator("#whoAmI").inner_text():
                fails.append("alice not shown")

            page.locator(f"#fontList li[data-stem='{drop_stem}']").click()
            page.locator('input[name="dec"][value="drop"]').click()
            page.wait_for_timeout(600)
            pt = progress(page)
            if "drop 1" not in pt or "已审 1 / 649" not in pt:
                fails.append(f"after drop progress: {pt}")
            disk = wait_disk_n(1)
            rec = disk["decisions"][drop_stem]
            if rec.get("review_decision") != "drop":
                fails.append(f"disk drop rec {rec}")
            if rec.get("reviewer") != "alice":
                fails.append(f"drop reviewer {rec.get('reviewer')}")
            if rec.get("split") not in {"train", "val", "test"}:
                fails.append(f"drop missing split {rec}")

            # 2 pass another
            page.locator("#filter").select_option("all")
            stems = list_stems(page)
            pass_stem = next(s for s in stems if s != drop_stem)
            page.locator(f"#fontList li[data-stem='{pass_stem}']").click()
            page.locator('input[name="dec"][value="pass"]').click()
            page.wait_for_timeout(600)
            wait_disk_n(2)

            # 3 rerender a third
            stems = list_stems(page)
            rr_stem = next(s for s in stems if s not in {drop_stem, pass_stem})
            page.locator(f"#fontList li[data-stem='{rr_stem}']").click()
            page.locator('input[name="dec"][value="rerender"]').click()
            page.wait_for_timeout(600)
            wait_disk_n(3)
            pt = progress(page)
            if "pass 1" not in pt or "drop 1" not in pt or "rerender 1" not in pt:
                fails.append(f"counts after 3: {pt}")
            if "我 3" not in pt:
                fails.append(f"mine count after 3: {pt}")
            page.locator("#filter").select_option("mine")
            page.wait_for_timeout(200)
            if set(list_stems(page)) != {drop_stem, pass_stem, rr_stem}:
                fails.append(f"filter mine {list_stems(page)}")
            alice_disk = wait_disk_n(3)
            alice_cursor = (alice_disk.get("reviewers") or {}).get("alice", {}).get("last_stem")
            if not alice_cursor:
                fails.append(f"alice last_stem missing {alice_disk.get('reviewers')}")

            # filters
            page.locator("#filter").select_option("drop")
            page.wait_for_timeout(200)
            got = list_stems(page)
            if got != [drop_stem]:
                fails.append(f"filter drop {got}")
            page.locator("#filter").select_option("pass")
            page.wait_for_timeout(200)
            got = list_stems(page)
            if got != [pass_stem]:
                fails.append(f"filter pass {got}")
            page.locator("#filter").select_option("rerender")
            page.wait_for_timeout(200)
            got = list_stems(page)
            if got != [rr_stem]:
                fails.append(f"filter rerender {got}")
            page.locator("#filter").select_option("undecided")
            page.wait_for_timeout(200)
            if len(list_stems(page)) != 646:
                fails.append(f"undecided {len(list_stems(page))}")
            page.locator("#filter").select_option("decided")
            page.wait_for_timeout(200)
            if set(list_stems(page)) != {drop_stem, pass_stem, rr_stem}:
                fails.append(f"decided {list_stems(page)}")

            page.locator("#filter").select_option("new389")
            page.wait_for_timeout(200)
            n389 = len(list_stems(page))
            page.locator("#filter").select_option("overlap260")
            page.wait_for_timeout(200)
            n260 = len(list_stems(page))
            if n389 + n260 != 649:
                fails.append(f"subset split {n389}+{n260}")

            page.locator("#filter").select_option("all")
            page.fill("#q", drop_stem[:8])
            page.wait_for_timeout(200)
            qstems = list_stems(page)
            if drop_stem not in qstems:
                fails.append(f"search missed {drop_stem} in {qstems[:5]}")

            # export jsonl
            page.fill("#q", "")
            page.locator("#filter").select_option("all")
            with page.expect_download() as dl:
                page.locator("#btnExport").click()
            download = dl.value
            out = REVIEW / "_ui_test_export.jsonl"
            download.save_as(str(out))
            lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
            by = {x["stem"]: x["review_decision"] for x in lines}
            expect = {drop_stem: "drop", pass_stem: "pass", rr_stem: "rerender"}
            if by != expect:
                fails.append(f"export jsonl {by} != {expect}")

            # reload same context: localStorage
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("#fontList li[data-stem]")
            page.wait_for_timeout(400)
            pt = progress(page)
            if "已审 3 / 649" not in pt:
                fails.append(f"reload local progress {pt}")
            if "alice" not in page.locator("#whoAmI").inner_text():
                fails.append(f"reload lost identity {page.locator('#whoAmI').inner_text()}")
            page.locator("#filter").select_option("drop")
            page.wait_for_timeout(200)
            if list_stems(page) != [drop_stem]:
                fails.append(f"reload filter drop {list_stems(page)}")

            # new context: disk only (no localStorage)
            ctx2 = browser.new_context()
            page2 = ctx2.new_page()
            page2.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page2.wait_for_selector("#fontList li[data-stem]")
            page2.wait_for_timeout(500)
            pt2 = progress(page2)
            if "已审 3 / 649" not in pt2:
                fails.append(f"fresh browser disk resume {pt2}")
            if "他人 3" not in pt2:
                fails.append(f"unsigned sees others: {pt2}")
            page2.locator("#filter").select_option("pass")
            page2.wait_for_timeout(200)
            if list_stems(page2) != [pass_stem]:
                fails.append(f"fresh filter pass {list_stems(page2)}")

            before_bob = json.loads(DEC_JSON.read_text(encoding="utf-8"))
            alice_cursor = (before_bob.get("reviewers") or {}).get("alice", {}).get("last_stem")
            sign_in(page2, "bob")
            pt_bob = progress(page2)
            if "已审 0 / 649" not in pt_bob:
                fails.append(f"bob progress should be his own: {pt_bob}")
            if "他人 3" not in pt_bob:
                fails.append(f"bob should see alice as others: {pt_bob}")
            page2.locator("#filter").select_option("mine")
            page2.wait_for_timeout(200)
            if list_stems(page2):
                fails.append(f"bob mine should be empty {list_stems(page2)}")
            page2.locator("#filter").select_option("others")
            page2.wait_for_timeout(200)
            if set(list_stems(page2)) != {drop_stem, pass_stem, rr_stem}:
                fails.append(f"bob others {list_stems(page2)}")
            page2.locator("#filter").select_option("all")
            page2.locator(f"#fontList li[data-stem='{drop_stem}']").click()
            page2.once("dialog", lambda dlg: dlg.dismiss())
            page2.locator('input[name="dec"][value="pass"]').click()
            page2.wait_for_timeout(500)
            after_bob = json.loads(DEC_JSON.read_text(encoding="utf-8"))
            if after_bob["decisions"][drop_stem].get("reviewer") != "alice":
                fails.append(f"overwrite without confirm {after_bob['decisions'][drop_stem]}")
            if (after_bob.get("reviewers") or {}).get("alice", {}).get("last_stem") != alice_cursor:
                fails.append(f"bob clobbered alice cursor {after_bob.get('reviewers')}")
            ctx2.close()
            ctx.close()

            # offline: same tab keeps going; other browser cannot see unsynced; reconnect flushes
            empty_disk()
            ctx4 = browser.new_context()
            page4 = ctx4.new_page()
            page4.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page4.wait_for_selector("#fontList li[data-stem]")
            sign_in(page4, "carol")
            stems4 = list_stems(page4)
            off0, off1, off2 = stems4[0], stems4[1], stems4[2]
            page4.locator(f"#fontList li[data-stem='{off0}']").click()
            page4.locator('input[name="dec"][value="drop"]').click()
            wait_disk_n(1)
            page4.context.set_offline(True)
            page4.locator(f"#fontList li[data-stem='{off1}']").click()
            page4.locator('input[name="dec"][value="pass"]').click()
            page4.wait_for_timeout(200)
            page4.locator(f"#fontList li[data-stem='{off2}']").click()
            page4.locator('input[name="dec"][value="rerender"]').click()
            page4.wait_for_timeout(700)
            disk_off = json.loads(DEC_JSON.read_text(encoding="utf-8"))
            if disk_off.get("n") != 1:
                fails.append(f"offline wrote disk n={disk_off.get('n')}")
            if "已审 3 / 649" not in progress(page4):
                fails.append(f"offline local progress {progress(page4)}")
            sync4 = page4.locator("#sync").inner_text()
            if "本机" not in sync4 and "中断" not in sync4 and "网络" not in sync4:
                fails.append(f"offline sync hint {sync4}")
            ctx5 = browser.new_context()
            page5 = ctx5.new_page()
            page5.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page5.wait_for_selector("#fontList li[data-stem]")
            page5.wait_for_timeout(400)
            if "已审 1 / 649" not in progress(page5):
                fails.append(f"other browser saw unsynced {progress(page5)}")
            ctx5.close()
            page4.context.set_offline(False)
            page4.evaluate("() => window.dispatchEvent(new Event('online'))")
            wait_disk_n(3)
            page4.reload(wait_until="domcontentloaded")
            page4.wait_for_selector("#fontList li[data-stem]")
            page4.wait_for_timeout(500)
            if "carol" not in page4.locator("#whoAmI").inner_text():
                fails.append(f"reconnect lost identity {page4.locator('#whoAmI').inner_text()}")
            if "已审 3 / 649" not in progress(page4):
                fails.append(f"reconnect resume {progress(page4)}")
            page4.locator("#filter").select_option("mine")
            page4.wait_for_timeout(200)
            if set(list_stems(page4)) != {off0, off1, off2}:
                fails.append(f"reconnect mine {list_stems(page4)}")
            ctx4.close()

            # 下次打开：断网后未点重试，联网 reload 也应把本机补写到磁盘
            empty_disk()
            ctx6 = browser.new_context()
            page6 = ctx6.new_page()
            page6.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page6.wait_for_selector("#fontList li[data-stem]")
            sign_in(page6, "dave")
            stems6 = list_stems(page6)
            page6.locator(f"#fontList li[data-stem='{stems6[0]}']").click()
            page6.locator('input[name="dec"][value="pass"]').click()
            wait_disk_n(1)
            page6.context.set_offline(True)
            page6.locator("#filter").select_option("all")
            page6.locator(f"#fontList li[data-stem='{stems6[1]}']").click()
            page6.locator('input[name="dec"][value="drop"]').click()
            page6.wait_for_timeout(700)
            if json.loads(DEC_JSON.read_text(encoding="utf-8")).get("n") != 1:
                fails.append("reload-path leaked before reconnect")
            page6.context.set_offline(False)
            page6.reload(wait_until="domcontentloaded")
            page6.wait_for_selector("#fontList li[data-stem]")
            wait_disk_n(2)
            if "已审 2 / 649" not in progress(page6):
                fails.append(f"reload flush {progress(page6)}")
            ctx6.close()

            # import jsonl into empty session then merge
            empty_disk()
            ctx3 = browser.new_context()
            page3 = ctx3.new_page()
            page3.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page3.wait_for_selector("#fontList li[data-stem]")
            page3.wait_for_timeout(400)
            if "已审 0 / 649" not in progress(page3):
                fails.append(f"after wipe {progress(page3)}")
            page3.on("dialog", lambda dlg: dlg.accept())
            page3.locator("#fileImport").set_input_files(str(out))
            page3.wait_for_timeout(1000)
            if "已审 3 / 649" not in progress(page3):
                fails.append(f"after import {progress(page3)}")
            wait_disk_n(3)
            ctx3.close()
            browser.close()
            out.unlink(missing_ok=True)

        if fails:
            raise SystemExit("FAIL\n" + "\n".join(fails))
        print("PASS")
        print(f"  drop={drop_stem} pass={pass_stem} rerender={rr_stem}")
        print("  click/filter/export/reload/fresh-browser/import all ok")
    finally:
        restore(*bak)
        print("restored prior decisions.json")


if __name__ == "__main__":
    main()
