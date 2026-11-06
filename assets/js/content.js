/*
  Site content — edit this file to update the portfolio.
  Every text field has an English (en) and French (fr) version.
  Source: CV "Q4 2026 - CV StarClay - Hantz T."
*/
window.SITE = {
  // GitHub username for the "Latest on GitHub" section. Leave empty to hide the section.
  // Example: "hp8990"
  githubUser: "",
  // Repositories to show when the GitHub section is on. Only these appear, in this order.
  // Leave empty to show none. Example: ["DataScience", "coding-challenge-"]
  githubRepos: [],
  // DEV (dev.to) username: the Writing section loads the latest articles live
  devtoUser: "hantz_peterthermoris_f64",
  // Leave empty to hide the email link. Example: "prenom.nom@example.com"
  email: "",
  links: {
    linkedin: "https://www.linkedin.com/in/hantz-peter-b-thermoris/",
    // Leave a link empty to hide it from the contact section.
    // Examples: "https://github.com/hp8990", "https://www.kaggle.com/thermorishantzpeter"
    github: "",
    kaggle: "",
    devto: "https://dev.to/hantz_peterthermoris_f64",
    // CV PDF for each page language. Leave both empty to hide the CV buttons.
    cv: {
      en: "docs/Hantz-Peter-Thermoris-CV-EN.pdf",
      fr: "docs/Hantz-Peter-Thermoris-CV-FR.pdf"
    }
  },
  // Shown next to CV links for accessibility. Update if you replace the PDF.
  cvSize: { en: "74 KB", fr: "76 KB" }
};

/*
  Agent harness demo in the introduction. A simulated run, steps in order.
  verdict: "allow" (runs automatically), "approval" (the visitor approves or rejects),
           "steer" (the harness rewrites the action into a safer one), "block" (refused).
*/
window.HARNESS = [
  {
    tool: "read_table(\"products\")",
    action: { en: "Read the product table", fr: "Lire la table des produits" },
    verdict: "allow",
    reason: { en: "Read-only tools run without review.", fr: "Les outils en lecture seule s'exécutent sans validation." },
    result: { en: "48,210 rows read.", fr: "48 210 lignes lues." }
  },
  {
    tool: "detect_outliers(column=\"price\", rule=\"IQR\")",
    action: { en: "Flag price outliers with the IQR rule", fr: "Repérer les prix aberrants avec la règle IQR" },
    verdict: "allow",
    reason: { en: "Analysis only: nothing is changed.", fr: "Analyse seule : rien n'est modifié." },
    result: { en: "37 outliers flagged.", fr: "37 valeurs aberrantes repérées." }
  },
  {
    tool: "fill_missing(column=\"colour\", source=\"supplier\")",
    action: { en: "Fill 312 missing colours from the supplier catalogue", fr: "Compléter 312 couleurs manquantes depuis le catalogue fournisseur" },
    verdict: "approval",
    reason: { en: "This action changes the data, so a person must approve it.", fr: "Cette action modifie les données : une personne doit la valider." },
    approved: { en: "312 rows updated.", fr: "312 lignes mises à jour." },
    rejected: { en: "Skipped. No data changed.", fr: "Ignorée. Aucune donnée modifiée." }
  },
  {
    tool: "delete_rows(filter=\"duplicates\")",
    action: { en: "Delete 1,204 suspected duplicates", fr: "Supprimer 1 204 doublons présumés" },
    verdict: "steer",
    steerTo: "flag_rows(filter=\"duplicates\")",
    reason: { en: "Deletion is not allowed. The harness turns it into flagging the rows for review.", fr: "La suppression est interdite. Le harness la transforme en signalement des lignes pour relecture." },
    result: { en: "1,204 rows flagged for review.", fr: "1 204 lignes signalées pour relecture." }
  },
  {
    tool: "send_email(to=\"client\")",
    action: { en: "Email the cleaned file to the client", fr: "Envoyer le fichier nettoyé au client par e-mail" },
    verdict: "block",
    reason: { en: "Agents may not contact people outside the team.", fr: "Les agents ne peuvent pas contacter de personnes extérieures à l'équipe." }
  }
];

/*
  "What I bring" section, shown right after the introduction.
  points: what working with me looks like. results: proof, each linked to its project.
  figureSr is what screen readers say instead of the figure, when the figure uses symbols.
*/
window.VALUE = {
  points: [
    {
      title: { en: "LLM systems that stay under control", fr: "Des systèmes LLM sous contrôle" },
      body: {
        en: "I design the agent harness around the models: tools exposed through MCP, human approval steps and full audit trails, so automation stays safe, traceable and GDPR-compliant.",
        fr: "Je conçois l'agent harness autour des modèles : outils exposés via MCP, étapes de validation humaine et journaux d'audit complets, pour une automatisation sûre, traçable et conforme au RGPD."
      }
    },
    {
      title: { en: "From business need to production", fr: "Du besoin métier à la production" },
      body: {
        en: "I run workshops with business teams, turn their requirements into detection scenarios and specifications, then deliver the pipeline, the model and the monitoring dashboards.",
        fr: "J'anime des ateliers avec les équipes métier, je traduis leurs besoins en scénarios de détection et en spécifications, puis je livre le pipeline, le modèle et les tableaux de bord de suivi."
      }
    },
    {
      title: { en: "Infrastructure fluency", fr: "Une maîtrise de l'infrastructure" },
      body: {
        en: "Six years of Linux, Windows Server, networking and databases: I deploy on Azure, AWS or on-premise without handing off the hard parts.",
        fr: "Six ans de Linux, Windows Server, réseaux et bases de données : je déploie sur Azure, AWS ou on-premise sans déléguer les parties difficiles."
      }
    },
    {
      title: { en: "Bilingual", fr: "Bilingue" },
      body: {
        en: "French native, English fluent (C1). I work with French and international teams alike.",
        fr: "Français langue maternelle, anglais courant (C1). Je travaille aussi bien avec des équipes françaises qu'internationales."
      }
    }
  ],
  results: [
    {
      figure: "+40%",
      figureFr: "+40\u00a0%",
      label: { en: "return over 7 months", fr: "de rendement sur 7 mois" },
      context: { en: "RAG decision engine for a trading group", fr: "Moteur de décision RAG pour un groupe de trading" },
      project: "trading-rag"
    },
    {
      figure: "~20%",
      figureFr: "~20\u00a0%",
      figureSr: { en: "nearly 20%", fr: "près de 20 %" },
      label: { en: "alert conversion on certain scenarios", fr: "de transformation des alertes sur certains scénarios" },
      context: { en: "Anomaly detection on billions of La Banque Postale transactions", fr: "Détection d'anomalies sur des milliards de transactions de La Banque Postale" },
      project: "lbp-anomaly"
    },
    {
      figure: "60% → 2%",
      figureFr: "60\u00a0% → 2\u00a0%",
      figureSr: { en: "from 60% to 2%", fr: "de 60 % à 2 %" },
      label: { en: "dropped calls", fr: "d'appels perdus" },
      context: { en: "After rebuilding a VoIP phone system", fr: "Après la refonte d'un système téléphonique VoIP" },
      project: "sogema-voip"
    }
  ]
};

/*
  Projects.
  track: "it" | "data" | "ai"   (used by the filters)
  kind:  "client" | "personal"  (shown as a small label)
  links: list of { label: {en, fr}, href }
*/
window.PROJECTS = [
  {
    id: "h-cleanser",
    track: "ai",
    kind: "client",
    year: "2026",
    org: "StarClay",
    title: { en: "Multi-agent data cleaning platform", fr: "Plateforme multi-agents de nettoyage de données" },
    summary: {
      en: "H-Cleanser-ERP turns massive, messy luxury-sector datasets into reliable structured data. Four LLM agents (explorer, analyst, enricher, curator) run on LangGraph with Gemini or local models through Ollama, and an agent harness keeps them in check: a custom MCP server exposing 11 data tools, a persistent human approval step and a full audit trail.",
      fr: "H-Cleanser-ERP transforme des jeux de données massifs et hétérogènes du secteur du luxe en données structurées et fiables. Quatre agents LLM (explorer, analyst, enricher, curator) tournent sur LangGraph avec Gemini ou des modèles locaux via Ollama, et un agent harness les encadre : serveur MCP exposant 11 outils de données, validation humaine persistante et journal d'audit complet."
    },
    tags: ["LangGraph", "Agent harness", "Gemini", "MCP", "FastAPI", "Celery", "Polars", "PostgreSQL", "Docker"],
    links: []
  },
  {
    id: "trading-rag",
    track: "ai",
    kind: "client",
    year: "2025",
    org: { en: "Private trading & investment group", fr: "Groupe privé de trading et d'investissement" },
    title: { en: "RAG decision engine for trading", fr: "Moteur de décision RAG pour le trading" },
    summary: {
      en: "A retrieval-augmented system on Phi-3-mini via Amazon Bedrock that synthesises news, technical indicators (RSI, VWAP, MACD) and financial reports into decision support, with automated ingestion on AWS Lambda and stress tests against black-swan scenarios. Close to 40% performance over the period.",
      fr: "Un système RAG basé sur Phi-3-mini via Amazon Bedrock qui synthétise news, indicateurs techniques (RSI, VWAP, MACD) et rapports financiers en aide à la décision, avec ingestion automatisée sur AWS Lambda et stress tests face aux scénarios de cygne noir. Une performance de près de 40 % sur la période."
    },
    tags: ["RAG", "Amazon Bedrock", "Phi-3", "AWS Lambda", "Prompt engineering"],
    links: []
  },
  {
    id: "lbp-anomaly",
    track: "data",
    kind: "client",
    year: "2023–2025",
    org: { en: "Xpert Consulting for La Banque Postale", fr: "Xpert Consulting pour La Banque Postale" },
    title: { en: "Anomaly detection on billions of bank transactions", fr: "Détection d'anomalies sur des milliards de transactions" },
    summary: {
      en: "Unsupervised models (Isolation Forest, Bisecting KMeans) on Hadoop to strengthen banking compliance, with optimised PySpark recipes in Dataiku and monitoring dashboards. Over 20% of generated alerts converted, a significant gain in operational precision.",
      fr: "Modèles non supervisés (Isolation Forest, Bisecting KMeans) sur Hadoop pour renforcer la conformité bancaire, avec des recipes PySpark optimisées dans Dataiku et des dashboards de suivi. Un taux de transformation des alertes supérieur à 20 %."
    },
    tags: ["PySpark", "Hadoop", "Isolation Forest", "Dataiku", "SQL"],
    links: []
  },
  {
    id: "reflexe-realestate",
    track: "data",
    kind: "client",
    year: "2023",
    org: "Reflexe",
    title: { en: "Real-estate chatbot and recommender", fr: "Chatbot et recommandation immobilière" },
    summary: {
      en: "An NLP support chatbot that answers common questions and qualifies leads, a user-centric property recommender built from browsing behaviour, and automated market data pipelines feeding pricing dashboards.",
      fr: "Un chatbot NLP qui répond aux questions fréquentes et qualifie les leads, un moteur de recommandation de biens basé sur les parcours de navigation, et des pipelines de collecte de données de marché alimentant des dashboards de prix."
    },
    tags: ["NLP", "BERT", "Recommendation", "Web scraping", "Random Forest"],
    links: []
  },
  {
    id: "capgemini-rgaa",
    track: "data",
    kind: "client",
    year: "2022",
    org: "Capgemini",
    title: { en: "Automated web accessibility audit", fr: "Audit automatisé de l'accessibilité web" },
    summary: {
      en: "An R&D project turning French RGAA accessibility criteria into algorithmic rules that analyse page structure, with Selenium and BeautifulSoup pipelines for single-page applications, from data collection through modelling.",
      fr: "Un projet R&D qui traduit les critères RGAA en règles algorithmiques d'analyse de la structure des pages, avec des pipelines Selenium et BeautifulSoup pour les applications monopages, de la collecte à la modélisation."
    },
    tags: ["Python", "Selenium", "BeautifulSoup", "Streamlit", "Power BI"],
    links: []
  },
  {
    id: "sogema-voip",
    track: "it",
    kind: "client",
    year: "2013–2019",
    org: "CRC Sogema (FIPCA)",
    title: { en: "Critical infrastructure for an international program", fr: "Infrastructure critique d'un programme international" },
    summary: {
      en: "Administered Linux and Windows Server 2012 R2, designed network and server infrastructure for high availability, and rebuilt the VoIP phone system, cutting dropped calls from 60% to 2%. Also built SQL project-tracking databases, small automation apps and level 2/3 support.",
      fr: "Administration de serveurs Linux et Windows Server 2012 R2, conception d'infrastructures réseau et serveurs à haute disponibilité, et refonte de la téléphonie VoIP qui a fait passer le taux d'appels perdus de 60 % à 2 %. Aussi : bases SQL de suivi de projet, micro-applications d'automatisation et support niveau 2/3."
    },
    tags: ["Linux", "Windows Server", "VoIP", "Networking", "SQL", "Bash"],
    links: []
  },
  {
    id: "nba",
    track: "data",
    kind: "personal",
    year: "",
    title: { en: "NBA game-log analysis", fr: "Analyse des matchs NBA" },
    summary: {
      en: "Pulling a player's full game history from the NBA stats API and analysing it with pandas, using Scottie Pippen's career as the case study.",
      fr: "Récupération de l'historique complet des matchs d'un joueur via l'API NBA et analyse avec pandas, avec la carrière de Scottie Pippen comme cas d'étude."
    },
    tags: ["Python", "pandas", "API"],
    links: [
      { label: { en: "Notebook", fr: "Notebook" }, href: "projects/nba_api.ipynb" },
      { label: { en: "Dataset (CSV)", fr: "Données (CSV)" }, href: "projects/data/scottie-pippen-career-stats.csv" }
    ]
  },
  {
    id: "sarcasm",
    track: "data",
    kind: "personal",
    year: "",
    title: { en: "Sarcasm detection dataset", fr: "Jeu de données de détection du sarcasme" },
    summary: {
      en: "Cleaning and preprocessing a news-headline corpus labelled sarcastic or not, ready for training an NLP classifier.",
      fr: "Nettoyage et prétraitement d'un corpus de titres de presse étiquetés sarcastiques ou non, prêt pour entraîner un classifieur NLP."
    },
    tags: ["Python", "NLP"],
    links: [{ label: { en: "Processed data", fr: "Données traitées" }, href: "projects/data/sarcasm-dataset/sarcasm_processed_data_v2.json" }]
  }
];

/* Timeline, newest first. */
window.TIMELINE = [
  {
    when: { en: "Apr 2026 – now", fr: "avr. 2026 – auj." },
    what: { en: "Data Scientist", fr: "Data Scientist" },
    where: { en: "StarClay", fr: "StarClay" }
  },
  {
    when: { en: "May – Nov 2025", fr: "mai – nov. 2025" },
    what: { en: "Data Scientist, trading & investment", fr: "Data Scientist, trading et investissement" },
    where: { en: "Private group", fr: "Groupe privé" }
  },
  {
    when: { en: "Oct 2023 – Mar 2025", fr: "oct. 2023 – mars 2025" },
    what: { en: "Data Scientist", fr: "Data Scientist" },
    where: { en: "Xpert Consulting, for La Banque Postale", fr: "Xpert Consulting, mission La Banque Postale" }
  },
  {
    when: { en: "Mar – Oct 2023", fr: "mars – oct. 2023" },
    what: { en: "Data Scientist, web specialist", fr: "Data Scientist, spécialiste web" },
    where: { en: "Reflexe (remote)", fr: "Reflexe (à distance)" }
  },
  {
    when: { en: "Mar – Sep 2022", fr: "mars – sept. 2022" },
    what: { en: "Data Scientist", fr: "Data Scientist" },
    where: { en: "Capgemini", fr: "Capgemini" }
  },
  {
    when: { en: "2022", fr: "2022" },
    what: { en: "Master's in AI & Data Science", fr: "Master Intelligence Artificielle & Science des Données" },
    where: { en: "IA School, France", fr: "IA School, France" },
    edu: true
  },
  {
    when: { en: "Nov 2013 – Aug 2019", fr: "nov. 2013 – août 2019" },
    what: { en: "Systems & networks engineer", fr: "Systèmes et réseaux" },
    where: { en: "CRC Sogema (FIPCA project) and SMEs", fr: "CRC Sogema (projet FIPCA) et PME" }
  },
  {
    when: { en: "2013", fr: "2013" },
    what: { en: "Master's in Electronic Engineering (Computing & Telecoms)", fr: "Master Génie Électronique (Informatique & Télécoms)" },
    where: { en: "State University of Haiti", fr: "Université d'État d'Haïti" },
    edu: true
  }
];

/*
  Writing hosted on this site. Always shown in the Writing section, next to the DEV articles.
  lang is the language the piece is written in; leave it out when title and url exist in both languages.
*/
window.WRITING_PINNED = [
  {
    // available in both languages: each page links to its own version
    title: { en: "Genealogy of Generative AI", fr: "Généalogie de l'IA générative" },
    url: { en: "projects/generative-ai-genealogy.html", fr: "projects/genealogie-ia-generative.html" },
    date: "2026-10-14",
    tags: ["ai", "research", "interactive"],
    summary: {
      en: "An interactive map of the research papers behind today's generative AI, with a note card for each paper and a chronological index.",
      fr: "Une carte interactive des articles de recherche à l'origine de l'IA générative actuelle, avec une fiche par article et un index chronologique."
    }
  }
];

/*
  Articles. Shown if DEV can't be reached; when it can, the latest
  articles load automatically and new posts appear without editing this.
*/
window.ARTICLES = [
  {
    title: "Hiring AI Engineers for AI Agents: Design Skills vs. Frameworks",
    url: "https://dev.to/hantz_peterthermoris_f64/hiring-ai-engineers-for-ai-agents-design-skills-vs-frameworks-2mj4",
    date: "2026-10-22",
    tags: ["ai", "agents", "softwareengineering", "career"],
    summary: "Why teams building AI agents should hire for design skills such as policy checks, human approval steps and audit logs, rather than familiarity with one framework."
  },
  {
    title: "Learn the Harness Concepts Before You Learn the Framework",
    url: "https://dev.to/hantz_peterthermoris_f64/learn-the-harness-concepts-before-you-learn-the-framework-12l2",
    date: "2026-09-16",
    tags: ["ai", "python", "programming", "productivity"],
    summary: "LangChain, Google ADK, Strands and CrewAI share the same agent control concepts under different names. Learn interception points, steering and skills once, and they carry across frameworks."
  }
];
