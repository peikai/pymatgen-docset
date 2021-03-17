## Guide 

1. Copy the *pymargen.docset* folder to docset storage directory of browsers, such as [Zeal](https://zealdocs.org/) or [Dash](https://kapeli.com/dash).

2. Restart the docset browser.

## Generation

1. clone the [pymatgen](https://github.com/materialsproject/pymatgen) repository.
2. move out tasks.py from root directory and adjust workdirs accordingly, then `invoke make-dash`.

