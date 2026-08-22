# Console web client

`console/static` is the browser application served by the Console. JavaScript
modules fetch API/panel payloads, render training and world views, stream live
job updates, and open run replays; CSS files provide the shared visual system.

It depends on the JSON contracts exposed by [`../api/`](../api/README.md) and
[`../panels/`](../panels/README.md). Static assets are served directly by the
Console and do not require a Python process restart for ordinary browser-side
changes.

Start with [`index.html`](index.html), [`js/main.js`](js/main.js),
[`js/train.js`](js/train.js), and [`js/runs.js`](js/runs.js).
