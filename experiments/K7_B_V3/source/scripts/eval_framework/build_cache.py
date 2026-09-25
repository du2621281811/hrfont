#!/usr/bin/env python3
"""命令行构建外部字体 cache。"""
from common import parse_cli
from data import build_cache, scan_fonts

def main(argv=None):
    c=parse_cli(argv); chars=list(c.get("chars","永和书风骨韵天地ABCDEFGHabcdefgh")); fonts=scan_fonts(c.get("fonts_dir"),c.get("fonts_manifest")); result=build_cache(fonts,chars,c["cache_dir"],int(c.get("canvas",96)),c.get("strategy","fixed_size"),int(c.get("fixed_size",80)),int(c.get("margin",6))); print(f"cache={c['cache_dir']} fonts={len(result['fonts'])} glyphs={len(result['records'])} sha256={result['cache_sha256']}")
if __name__=="__main__": main()
