/*
  Site content — edit this file to update the portfolio.
  Every text field has an English (en) and French (fr) version.
  Source: CV "Q4 2026 - CV StarClay - Hantz T."
*/
window.SITE = {
  githubUser: "hp8990",
  // Leave empty to hide the email link. Example: "prenom.nom@example.com"
  email: "",
  links: {
    linkedin: "https://www.linkedin.com/in/hantz-peter-b-thermoris/",
    github: "https://github.com/hp8990",
    kaggle: "https://www.kaggle.com/thermorishantzpeter",
    cv: "docs/Hantz-Peter-Thermoris-CV.pdf"
  },
  // Shown next to CV links for accessibility. Update if you replace the PDF.
  cvSize: "144 KB"
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
      en: "H-Cleanser-ERP turns massive, messy luxury-sector datasets into reliable structured data. Four LLM agents (explorer, analyst, enricher, curator) run on LangGraph with Gemini or local models through Ollama, a custom MCP server exposes 11 data tools, and every correction passes through a persistent human approval step with a full audit trail.",
      fr: "H-Cleanser-ERP transforme des jeux de données massifs et hétérogènes du secteur du luxe en données structurées et fiables. Quatre agents LLM (explorer, analyst, enricher, curator) tournent sur LangGraph avec Gemini ou des modèles locaux via Ollama, un serveur MCP expose 11 outils de données, et chaque correction passe par une validation humaine persistante avec journal d'audit complet."
    },
    tags: ["LangGraph", "Gemini", "MCP", "FastAPI", "Celery", "Polars", "PostgreSQL", "Docker"],
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
    id: "keystrokes",
    track: "ai",
    kind: "personal",
    year: "",
    title: { en: "Keystroke dynamics biometrics", fr: "Biométrie par dynamique de frappe" },
    summary: {
      en: "Identifying a typist from how they type, not what they type. I recorded key timings for the left hand, right hand and both hands, then compared hold time against the pause between keys.",
      fr: "Reconnaître une personne à sa façon de taper, et non à ce qu'elle tape. J'ai enregistré les temps de frappe main gauche, main droite et deux mains, puis comparé le temps d'appui au temps entre deux touches."
    },
    tags: ["Python", "Biometrics", "Matplotlib"],
    image: "assets/img/projects/keystrokes.jpg",
    imageAlt: { en: "Scatter plot of hold time and dwell time for each key pressed", fr: "Nuage de points des temps d'appui et de vol pour chaque touche" },
    links: []
  },
  {
    id: "facial-expression",
    track: "ai",
    kind: "personal",
    year: "",
    title: { en: "Facial expression recognition", fr: "Reconnaissance des expressions faciales" },
    summary: {
      en: "Recognising the seven universal facial expressions from a webcam feed, with data augmentation and transfer learning on VGG19.",
      fr: "Reconnaissance des sept expressions faciales universelles depuis une webcam, avec augmentation de données et transfer learning sur VGG19."
    },
    tags: ["TensorFlow", "Keras", "VGG19", "Computer vision"],
    image: "assets/img/projects/facial-expression.jpg",
    imageAlt: { en: "Webcam frame with a detected face labelled Surprise", fr: "Image webcam avec un visage détecté étiqueté Surprise" },
    links: []
  },
  {
    id: "iris-tracking",
    track: "ai",
    kind: "personal",
    year: "",
    title: { en: "Real-time iris tracking", fr: "Suivi de l'iris en temps réel" },
    summary: {
      en: "Webcam eye and iris tracking with OpenCV and Dlib, using the 68-point facial landmark model to follow the eyes frame by frame.",
      fr: "Suivi des yeux et de l'iris par webcam avec OpenCV et Dlib, à l'aide du modèle à 68 points de repère du visage."
    },
    tags: ["Python", "OpenCV", "Dlib"],
    links: [{ label: { en: "Source code", fr: "Code source" }, href: "projects/iris_tracking.py" }]
  },
  {
    id: "genai-genealogy",
    track: "ai",
    kind: "personal",
    year: "",
    title: { en: "Generative AI genealogy", fr: "Généalogie de l'IA générative" },
    summary: {
      en: "An interactive map of the research papers behind today's generative AI, with a note card for each paper and a chronological index.",
      fr: "Une carte interactive des articles de recherche à l'origine de l'IA générative, avec une fiche par article et un index chronologique."
    },
    tags: ["Generative AI", "JavaScript"],
    links: [{ label: { en: "Open the map", fr: "Ouvrir la carte" }, href: "projects/genealogie-ia-generative.html" }]
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
