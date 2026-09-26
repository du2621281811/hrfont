# K7-B filtered-donor 24-pair review

`k7_b_24pair_prefix_filtered_gallery.zip` is the complete visual review bundle for all 24 target font/glyph pairs. It contains the filtered-donor rerun, GT and original K7-B baseline, the ten retained donor glyphs and their eight references per donor, and the target font's eight input references. The original, unfiltered donor image folders are not included.

Extract the ZIP and serve its folder over local HTTP; open `prefix_review.html` from that server. The page uses `fetch`, so opening it directly as `file://` will fail. The extracted bundle's README has the protocol and viewing instructions.

Protocol: K7-B EMA 32K, 8-shot, seed 3407. Donor font names are filtered after punctuation removal when they share at least 8 consecutive characters and at least 60% of the longer name. The comparison is the original K7-B run versus the filtered-donor rerun.

The archive contains 2,424 PNG images across the 24 pairs. SHA-256 is recorded in `SHA256SUMS`.
