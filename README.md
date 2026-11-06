# hp8990.github.io

Personal portfolio of Hantz Peter B. Thermoris, data scientist and LLM/AI engineer with a background in systems and networks. Live at https://hp8990.github.io

## Structure

| Path | What it is |
| --- | --- |
| `index.html` | The homepage |
| `assets/js/content.js` | **All portfolio content**: projects, timeline, links. Edit this to update the site. |
| `assets/js/i18n.js` | Interface text in English and French |
| `assets/js/main.js` | Language and theme switch, project filters, live GitHub repos and DEV articles, keystroke demo |
| `assets/css/site.css` | Styles (light and dark themes) |
| `assets/img/` | Portrait and project images |
| `docs/` | Downloadable documents such as your CV (empty for now) |
| `projects/` | Project pages, notebooks, code and datasets |
| `archive/` | Previous versions of the site, kept for reference |
| `ms70602493.txt` | Microsoft 365 domain verification. Must stay at the root. |

## Updating

- **Add a project:** copy an entry in `PROJECTS` in `assets/js/content.js`, then set `track` to `it`, `data` or `ai` and `kind` to `client` or `personal`.
- **Articles:** the Writing section loads your latest posts from DEV automatically. `ARTICLES` in `content.js` is the backup list shown if DEV doesn't respond.
- **Add your CV:** the CV buttons are hidden while `cv` in `content.js` is empty. Put the PDF in `docs/`, then set `cv` to its path (for example `docs/Hantz-Peter-Thermoris-CV.pdf`) and `cvSize` to its file size.
- **Show an email address:** set `email` in `assets/js/content.js`.

The site is plain HTML, CSS and JavaScript with no build step, so GitHub Pages serves it as is. To preview locally, run `python3 -m http.server` in this folder and open http://localhost:8000.
