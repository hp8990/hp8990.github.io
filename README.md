# hp8990.github.io

The personal portfolio of **Hantz Peter B. Thermoris**, data scientist and LLM/AI engineer with six years of prior experience in systems and networks.

**Live site:** https://hp8990.github.io

## Goals

- **Show both sides of the profile.** Machine learning in production since 2022 and LLM systems since 2025 (RAG, multi-agent pipelines), next to the infrastructure background they grew from, for recruiters, hiring managers and clients.
- **Speak to French and international readers.** Every piece of text exists in English and French, and visitors can switch at any time.
- **Stay current with little effort.** All content lives in one file, and the latest DEV articles load on their own.
- **Be accessible.** The site follows WCAG 2.1 AA, the standard behind France's RGAA.
- **Stay simple.** Plain HTML, CSS and JavaScript, with no framework, no build step and no tracking.

## What's on the site

| Section | Contents |
| --- | --- |
| Introduction | Who Hantz is, plus a live typing-rhythm demo based on his keystroke biometrics project |
| What I do | Two tracks: systems, cloud & DevOps, and data science & AI, each with its main tools |
| Selected work | Client projects and personal projects, filterable by IT & infrastructure, data science, or AI & LLMs |
| Writing | Latest articles from [DEV](https://dev.to/hantz_peterthermoris_f64), loaded live |
| Latest on GitHub | Only the repositories you choose, loaded live. Hidden for now. |
| Path | Career and education timeline, certifications and languages |
| Contact | LinkedIn and DEV (GitHub and Kaggle can be switched on) |

Light and dark themes follow the visitor's system setting and can be switched manually. The language and theme choices are remembered in the visitor's browser.

## Repository layout

| Path | What it holds |
| --- | --- |
| `index.html` | The single page of the site |
| `assets/js/content.js` | **All portfolio content**: projects, timeline, articles, links and settings |
| `assets/js/i18n.js` | Interface text (menus, buttons, messages) in English and French |
| `assets/js/main.js` | Page behaviour: language and theme switch, project filters, GitHub and DEV loading, typing demo |
| `assets/css/site.css` | All styles, including the dark theme |
| `assets/img/` | Portrait and project images |
| `projects/` | Material linked from project entries: the generative AI genealogy map, the NBA analysis notebook, the iris-tracking script, and datasets in `projects/data/` |
| `docs/` | The public CV in English and French (PDF) |
| `archive/` | Earlier drafts of the site, kept for reference and not linked from the homepage |
| `ms70602493.txt` | Microsoft 365 domain verification file. It must stay at the root. |

## Updating the content

Almost every change happens in `assets/js/content.js`. Each text field has an `en` and an `fr` version, so fill in both.

- **Add a project:** copy an entry in `PROJECTS`. Set `track` to `it`, `data` or `ai` (this drives the filters) and `kind` to `client` or `personal`. An `image` is optional; if you add one, give it an `imageAlt` description in both languages.
- **Add a career step:** add an entry to `TIMELINE`, newest first. Mark education entries with `edu: true`.
- **Articles:** new DEV posts appear on their own. `ARTICLES` is only the backup list shown when DEV can't be reached.
- **Update the CV:** the site offers the CV in the page language: `docs/Hantz-Peter-Thermoris-CV-EN.pdf` on the English page and `docs/Hantz-Peter-Thermoris-CV-FR.pdf` on the French page. Replace a file under the same name and update its `cvSize`. Leave `cv` empty to hide the CV buttons.
- **Show an email address:** set `email`. It stays hidden while empty.
- **Show GitHub or Kaggle:** set `links.github` or `links.kaggle` to add them to the contact section, and `githubUser` plus `githubRepos` to show a "Latest on GitHub" section listing only the repositories you name. All of these are empty for now.
- **Change interface text** such as menu labels or button names in `assets/js/i18n.js`.

## Preview and publish

To preview locally, run this in the repository folder and open http://localhost:8000:

```
python3 -m http.server
```

To publish, push to `main`. GitHub Pages serves the repository as is, and changes go live within a minute or two.

## Accessibility

The site meets WCAG 2.1 AA, tested with axe-core in light and dark themes, in both languages, on desktop and mobile. That covers text and control contrast, full keyboard use with visible focus, screen reader labels, announcements for filters and new-tab links, reflow down to 320 px, 200% text size, reduced motion and Windows high-contrast mode.

When adding content, keep it that way:

- Give every image a meaningful `imageAlt` in both languages.
- Don't rely on color alone to carry meaning.
- Keep link text descriptive. "Source code" says more than "here".

## Privacy

There are no cookies, analytics or third-party trackers. The typing demo runs entirely in the visitor's browser, and nothing typed is sent anywhere. The browser stores only the chosen language and theme, plus a 30-minute cache of the DEV list (and the GitHub list when enabled). The page requests data only from Google Fonts and the DEV API, plus the GitHub API when that section is enabled.
