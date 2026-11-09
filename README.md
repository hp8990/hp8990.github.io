# Hantz Peter B. Thermoris

**Data Scientist and AI Engineer (LLM), based in France.**
Portfolio: https://hantzthermoris.com | [Version française](#version-française)

## Who I am

I build machine learning and LLM systems that hold up in production. Since 2022 I have delivered machine learning for clients in banking, trading, real estate and consulting. Since 2025 I have focused on LLM systems: RAG, multi-agent pipelines, and the agent harnesses that keep them under control.

Before data science, I spent six years running servers, networks and phone systems for an international program. That background shapes how I work: I think about reliability, security and operations from the first day of a project, not after it ships.

## The value I bring

- **Measurable results.** A RAG decision engine for a trading group returned +40% over 7 months. Anomaly detection on billions of La Banque Postale transactions reached an alert conversion rate of nearly 20% on certain scenarios. Rebuilding a VoIP system cut dropped calls from 60% to 2%.
- **LLM systems that stay under control.** I design the agent harness around the models: tools exposed through MCP, human approval steps and full audit trails, so automation stays safe, traceable and GDPR-compliant.
- **From business need to production.** I run workshops with business teams, turn their requirements into detection scenarios and specifications, then deliver the pipeline, the model and the monitoring dashboards.
- **Infrastructure fluency.** Years of Linux, Windows Server, networking and databases mean I can deploy on Azure, AWS or on-premise without handing off the hard parts.
- **Bilingual.** French native, English fluent (C1).

**Certifications:** AWS Certified AI Practitioner; Microsoft Certified: Azure Data Scientist Associate, Power BI Data Analyst Associate, Azure Fundamentals, Azure AI Fundamentals.

**Find me:** [LinkedIn](https://www.linkedin.com/in/hantz-peter-b-thermoris/) | [Articles on DEV](https://dev.to/hantz_peterthermoris_f64) | CV in [English](https://hantzthermoris.com/docs/Hantz-Peter-Thermoris-CV-EN.pdf) or [French](https://hantzthermoris.com/docs/Hantz-Peter-Thermoris-CV-FR.pdf)

---

## Version française

**Data Scientist et Ingénieur IA (LLM), basé en France.**

### Qui je suis

Je conçois des systèmes de machine learning et de LLM qui tiennent en production. Depuis 2022, je livre du machine learning pour des clients de la banque, du trading, de l'immobilier et du conseil. Depuis 2025, je me concentre sur les systèmes LLM : RAG, pipelines multi-agents et agent harness qui les maintiennent sous contrôle.

Avant la data science, j'ai passé six ans à administrer des serveurs, des réseaux et des systèmes téléphoniques pour un programme international. Ce parcours guide ma façon de travailler : je pense fiabilité, sécurité et exploitation dès le premier jour d'un projet, pas après sa mise en ligne.

### Ce que j'apporte

- **Des résultats mesurables.** Un moteur de décision RAG pour un groupe de trading a généré +40 % de rendement sur 7 mois. La détection d'anomalies sur des milliards de transactions de La Banque Postale a atteint près de 20 % de transformation des alertes sur certains scénarios. La refonte d'un système VoIP a fait passer les appels perdus de 60 % à 2 %.
- **Des systèmes LLM sous contrôle.** Je conçois l'agent harness autour des modèles : outils exposés via MCP, étapes de validation humaine et journaux d'audit complets, pour une automatisation sûre, traçable et conforme au RGPD.
- **Du besoin métier à la production.** J'anime des ateliers avec les équipes métier, je traduis leurs besoins en scénarios de détection et en spécifications, puis je livre le pipeline, le modèle et les tableaux de bord de suivi.
- **Une maîtrise de l'infrastructure.** Des années de Linux, Windows Server, réseaux et bases de données me permettent de déployer sur Azure, AWS ou on-premise sans déléguer les parties difficiles.
- **Bilingue.** Français langue maternelle, anglais courant (C1).

**Certifications :** AWS Certified AI Practitioner ; Microsoft Certified : Azure Data Scientist Associate, Power BI Data Analyst Associate, Azure Fundamentals, Azure AI Fundamentals.

**Me retrouver :** [LinkedIn](https://www.linkedin.com/in/hantz-peter-b-thermoris/) | [Articles sur DEV](https://dev.to/hantz_peterthermoris_f64) | CV en [français](https://hantzthermoris.com/docs/Hantz-Peter-Thermoris-CV-FR.pdf) ou en [anglais](https://hantzthermoris.com/docs/Hantz-Peter-Thermoris-CV-EN.pdf)

---

# About this repository

This repository is the source of the portfolio site, served by GitHub Pages at https://hantzthermoris.com (the old address, hp8990.github.io, redirects there).

## Goals of the site

- **Show both sides of the profile.** Machine learning in production since 2022 and LLM systems since 2025 (RAG, multi-agent pipelines), next to the infrastructure background they grew from, for recruiters, hiring managers and clients.
- **Speak to French and international readers.** Every piece of text exists in English and French, and visitors can switch at any time.
- **Stay current with little effort.** All content lives in one file, and the latest DEV articles load on their own.
- **Be accessible.** The site follows WCAG 2.1 AA, the standard behind France's RGAA.
- **Stay simple.** Plain HTML, CSS and JavaScript, with no framework, no build step and no tracking.

## What's on the site

| Section | Contents |
| --- | --- |
| Introduction | Who Hantz is, plus an interactive agent harness demo: a simulated agent whose actions are allowed, sent for the visitor's approval, steered or blocked, with an audit log |
| What I bring | Four strengths and three measured results, each linked to its project |
| What I do | Two tracks: systems, cloud & DevOps, and data science & AI, each with its main tools |
| Selected work | Client projects, filterable by IT & infrastructure, data science, or AI & LLMs, with personal projects listed below under Side projects |
| Writing | The generative AI genealogy map, plus the latest articles from [DEV](https://dev.to/hantz_peterthermoris_f64), loaded live |
| Latest on GitHub | Only the repositories you choose, loaded live. Hidden for now. |
| Path | Career and education timeline, certifications and languages |
| Contact | LinkedIn and DEV (GitHub and Kaggle can be switched on) |

Light and dark themes follow the visitor's system setting and can be switched manually. The language and theme choices are remembered in the visitor's browser.

## Repository layout

| Path | What it holds |
| --- | --- |
| `index.html` | The single page of the site |
| `assets/js/content.js` | **All portfolio content**: strengths and results, projects, timeline, articles, links and settings |
| `assets/js/i18n.js` | Interface text (menus, buttons, messages) in English and French |
| `assets/js/main.js` | Page behaviour: language and theme switch, project filters, GitHub and DEV loading, agent harness demo |
| `assets/css/site.css` | All styles, including the dark theme |
| `assets/img/` | Portrait and project images |
| `projects/` | Material linked from the site: the generative AI genealogy map in English and French (shown under Writing), the NBA analysis notebook, and datasets in `projects/data/` |
| `docs/` | The public CV in English and French (PDF) |
| `archive/` | Earlier drafts of the site, kept for reference and not linked from the homepage |
| `CNAME` | Tells GitHub Pages to serve the site at hantzthermoris.com. Don't delete it. |
| `ms70602493.txt` | Microsoft 365 domain verification file. It must stay at the root. |

## Updating the content

Almost every change happens in `assets/js/content.js`. Each text field has an `en` and an `fr` version, so fill in both.

- **Agent harness demo:** the steps of the simulated run are in `HARNESS` in `content.js`. Each has a tool call, an action and a verdict (`allow`, `approval`, `steer` or `block`) with its explanation.
- **What I bring:** edit `VALUE` in `content.js`. `points` are the strengths; `results` are the figures, each with a short label, a context line and the `id` of the project it links to.
- **Add a project:** copy an entry in `PROJECTS`. Set `track` to `it`, `data` or `ai` (this drives the filters) and `kind` to `client` or `personal` (personal projects appear under Side projects). An `image` is optional; if you add one, give it an `imageAlt` description in both languages.
- **Add a career step:** add an entry to `TIMELINE`, newest first. Mark education entries with `edu: true`.
- **Writing on this site:** pieces hosted here, like the genealogy map, go in `WRITING_PINNED` in `content.js` and always show in the Writing section.
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

There are no cookies, analytics or third-party trackers. The agent harness demo is a simulation that runs entirely in the visitor's browser. The browser stores only the chosen language and theme, plus a 30-minute cache of the DEV list (and the GitHub list when enabled). The page requests data only from Google Fonts and the DEV API, plus the GitHub API when that section is enabled.
