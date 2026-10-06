# Typefaces

The page is set in two typefaces. Both are licensed under the
[SIL Open Font License, Version 1.1](https://openfontlicense.org), and both
files carry that licence in their own `name` table, where any font viewer
shows it. `redlight build-site` inlines them into `docs/index.html` as data
URIs, so the published page makes no font request of its own.

| File | Typeface | Copyright | What is in it |
|---|---|---|---|
| `black-han-sans.woff2` | Black Han Sans | Copyright 2015 The Black Han Sans Project Authors (https://github.com/zesstype/Black-Han-Sans) | A subset: printable ASCII, the Latin-1 and punctuation glyphs the font has, and the ten Hangul syllables of the chant, 무궁화 꽃이 피었습니다. Made with `pyftsubset` from the Google Fonts release, licence fields kept. |
| `ibm-plex-sans-kr-400.woff2` | IBM Plex Sans KR, Regular | Copyright 2018 IBM Corp., with Reserved Font Name "Plex" | The Latin subset as published by Fontsource (`@fontsource/ibm-plex-sans-kr` 5.3.0), unmodified. |
| `ibm-plex-sans-kr-600.woff2` | IBM Plex Sans KR, SemiBold | Copyright 2018 IBM Corp., with Reserved Font Name "Plex" | The Latin subset as published by Fontsource (`@fontsource/ibm-plex-sans-kr` 5.3.0), unmodified. |

The Plex files are redistributed exactly as published rather than subset
again here, because the Open Font License reserves the name "Plex" for
unmodified copies.
