# MAX78000 MMS Presentation

Build the 14-slide ETH Beamer deck with:

```sh
latexmk -pdf presentation.tex
```

All requested figures use compile-safe placeholders. Add images under `assets/` with the exact
filenames listed in `assets/README.md`; recompiling replaces each placeholder automatically.

The title, author, institute and date are defined at the top of `presentation.tex`.
