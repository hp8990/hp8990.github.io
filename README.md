# hp8990.github.io

Personal portfolio of Hantz Peter B. Thermoris, data scientist and LLM/AI engineer with a background in systems and networks. Live at https://hp8990.github.io

## Structure

| Path | What it is |
| --- | --- |
| `index.html` | The homepage |
| `assets/js/content.js` | **All portfolio content**: projects, timeline, links. Edit this to update the site. |
| `assets/js/i18n.js` | Interface text in English and French |
| `assets/js/main.js` | Language and theme switch, project filters, live GitHub repos, keystroke demo |
| `assets/css/site.css` | Styles (light and dark themes) |
| `assets/img/` | Portrait and project images |
| `docs/` | CV (PDF) |
| `projects/` | Project pages, notebooks, code and datasets |
| `archive/` | Previous versions of the site, kept for reference |
| `ms70602493.txt` | Microsoft 365 domain verification. Must stay at the root. |

## Updating

- **Add a project:** copy an entry in `PROJECTS` in `assets/js/content.js`, then set `track` to `it`, `data` or `ai` and `kind` to `client` or `personal`.
- **Replace the CV:** overwrite `docs/Hantz-Peter-Thermoris-CV.pdf`, keeping the same name.
- **Show an email address:** set `email` in `assets/js/content.js`.

The site is plain HTML, CSS and JavaScript with no build step, so GitHub Pages serves it as is. To preview locally, run `python3 -m http.server` in this folder and open http://localhost:8000.
