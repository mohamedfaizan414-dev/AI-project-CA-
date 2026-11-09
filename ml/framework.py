"""
framework.py  --  THE COMPETENCY FRAMEWORK
===========================================
This file answers two questions:
  1. WHAT skills (competencies) exist for people in India's Official Statistical System?
  2. HOW GOOD must each job role be at every skill (1 = beginner ... 5 = expert)?

Why we wrote it by hand: iGOT Karmayogi's real competency data is not public.
So we follow the same *shape* iGOT uses (Behavioural / Functional / Domain
competencies, each with proficiency levels) and fill it with skills that
statistical officers actually need. If MoSPI later gives the real list, you only
replace this one file and everything else in the project re-runs automatically.
"""

# --------------------------------------------------------------------------
# PART 1: the competencies.
# Each line:  (id, nice name, group, one-line description, [keywords])
# - "keywords" are words that, when found in a course title/description,
#   suggest that the course teaches this competency. They are the starting
#   point ("weak supervision") for the course-tagging AI in 03_tag_courses.py.
# --------------------------------------------------------------------------
_RAW = [
    # ---------------- DOMAIN: knowledge specific to official statistics -------------
    ("survey_design", "Survey Design & Sampling", "Domain",
     "Design questionnaires and sample designs, run censuses and household surveys.",
     ["survey", "sampling", "questionnaire", "sample design", "census", "data collection", "polling"]),
    ("stat_inference", "Statistical Inference", "Domain",
     "Use probability, estimation and hypothesis tests to draw sound conclusions from data.",
     ["statistics", "statistical", "inference", "hypothesis testing", "probability", "confidence interval", "bayesian", "anova", "distribution"]),
    ("regression", "Regression & Econometrics", "Domain",
     "Model relationships between variables with regression and causal methods.",
     ["regression", "econometrics", "linear model", "causal inference", "logistic regression", "predictive modeling"]),
    ("time_series", "Time Series & Forecasting", "Domain",
     "Analyse data over time: trends, seasonality, forecasting of indicators.",
     ["time series", "forecasting", "arima", "seasonal", "forecast"]),
    ("macro_indicators", "Economic Indicators & National Accounts", "Domain",
     "Understand GDP, inflation, price indices and how national accounts are built.",
     ["macroeconomics", "economics", "gdp", "national accounts", "inflation", "economic indicators", "microeconomics", "development economics"]),
    ("official_quality", "Data Quality & Statistical Standards", "Domain",
     "Apply quality frameworks, metadata and international standards to official data.",
     ["data quality", "quality management", "quality assurance", "metadata", "statistical standards", "international standards", "sdmx", "six sigma", "statistical process"]),
    ("geospatial", "Geospatial Analysis & GIS", "Domain",
     "Work with maps, spatial data and remote sensing for statistics.",
     ["gis", "geospatial", "remote sensing", "mapping", "spatial analysis", "geographic", "arcgis", "qgis"]),
    ("research_methods", "Research & Evaluation Methods", "Domain",
     "Plan studies, evaluate programmes and design experiments.",
     ["research methods", "evaluation", "qualitative", "experimental design", "impact evaluation", "a/b testing", "scientific method", "research design"]),

    # ---------------- FUNCTIONAL: tools and technical skills -------------------------
    ("data_wrangling", "Data Cleaning & Wrangling", "Functional",
     "Clean, reshape and combine messy data before analysis.",
     ["data cleaning", "wrangling", "preprocessing", "data preparation", "pandas", "data manipulation", "etl", "exploratory data analysis", "data processing", "data transformation", "data munging", "tidy data"]),
    ("spreadsheets", "Spreadsheet Analytics", "Functional",
     "Use spreadsheets for tabulation, pivots and quick analysis.",
     ["excel", "spreadsheet", "google sheets", "pivot", "vba"]),
    ("sql_db", "Databases & SQL", "Functional",
     "Store and query data with relational databases.",
     ["sql", "database", "mysql", "postgresql", "nosql", "relational", "mongodb", "data warehouse"]),
    ("python_analysis", "Python for Data Analysis", "Functional",
     "Analyse data with Python, pandas and notebooks.",
     ["python", "pandas", "numpy", "jupyter", "scipy"]),
    ("r_analysis", "R Programming", "Functional",
     "Analyse and plot data in R, the statistician's language.",
     ["r programming", "rstudio", "tidyverse", "ggplot", "r language", "shiny"]),
    ("data_viz", "Data Visualisation & Dashboards", "Functional",
     "Turn numbers into clear charts and interactive dashboards.",
     ["data visualization", "visualisation", "visualization", "tableau", "power bi", "dashboard", "charts"]),
    ("machine_learning", "Machine Learning", "Functional",
     "Build models that learn patterns: classification, clustering, prediction.",
     ["machine learning", "supervised", "unsupervised", "neural network", "scikit", "classification", "clustering", "random forest"]),
    ("deep_learning_ai", "AI & Deep Learning Awareness", "Functional",
     "Understand modern AI, deep learning and generative models and their limits.",
     ["deep learning", "artificial intelligence", "tensorflow", "pytorch", "generative ai", "llm", "computer vision"]),
    ("text_nlp", "Text Mining & NLP", "Functional",
     "Extract information from documents, open-ended survey answers and text.",
     ["nlp", "natural language", "text mining", "text analytics", "sentiment"]),
    ("big_data_cloud", "Big Data & Cloud", "Functional",
     "Handle very large datasets with cloud and distributed tools.",
     ["big data", "hadoop", "spark", "cloud computing", "aws", "azure", "google cloud", "data engineering"]),
    ("data_governance", "Data Governance, Privacy & Security", "Functional",
     "Protect confidentiality, follow data-protection rules, manage data responsibly.",
     ["data governance", "privacy", "cybersecurity", "security", "confidentiality", "gdpr", "data protection", "cryptography"]),
    ("quant_foundations", "Quantitative & Maths Foundations", "Functional",
     "Algebra, calculus and numeracy that all statistics is built on.",
     ["calculus", "linear algebra", "mathematics", "algebra", "mathematical", "numeracy", "discrete math"]),

    # ---------------- BEHAVIOURAL: how you work with people --------------------------
    ("report_writing", "Report Writing & Dissemination", "Behavioural",
     "Write clear statistical reports, press notes and technical documents.",
     ["writing", "report writing", "technical writing", "business writing", "editing", "academic writing"]),
    ("data_storytelling", "Data Storytelling & Presentation", "Behavioural",
     "Explain findings so decision makers act on them.",
     ["storytelling", "presentation", "public speaking", "presenting"]),
    ("communication", "Communication & Stakeholder Engagement", "Behavioural",
     "Listen, negotiate and work with data users, respondents and ministries.",
     ["communication", "negotiation", "stakeholder", "influence", "interpersonal", "conflict"]),
    ("leadership", "Leadership & People Management", "Behavioural",
     "Lead teams, coach juniors and manage performance.",
     ["leadership", "leading teams", "people management", "coaching", "management skills", "manager"]),
    ("project_mgmt", "Project & Programme Management", "Behavioural",
     "Plan, schedule and deliver statistical projects on time.",
     ["project management", "agile", "scrum", "pmp", "program management", "planning"]),
    ("policy_analysis", "Public Policy & Governance", "Behavioural",
     "Connect statistics to evidence-based policy and public administration.",
     ["public policy", "policy", "governance", "public administration", "government", "public sector", "sustainable development"]),
    ("critical_thinking", "Critical Thinking & Problem Solving", "Behavioural",
     "Question assumptions, reason with evidence and solve open problems.",
     ["critical thinking", "problem solving", "decision making", "reasoning", "analytical thinking"]),
    ("ethics_integrity", "Ethics & Integrity in Public Service", "Behavioural",
     "Act with integrity, impartiality and professional ethics.",
     ["ethics", "integrity", "ethical", "corporate social responsibility", "compliance"]),
    ("change_innovation", "Change & Innovation Management", "Behavioural",
     "Lead adoption of new methods, tools and digital transformation.",
     ["change management", "innovation", "design thinking", "digital transformation", "creativity"]),
    ("productivity_learning", "Productivity & Learning Skills", "Behavioural",
     "Manage time, stay effective and keep learning.",
     ["time management", "productivity", "learning how to learn", "study skills", "personal effectiveness", "resilience"]),
    ("public_finance", "Finance, Budgeting & Accounting", "Behavioural",
     "Understand budgets, accounting and expenditure data.",
     ["finance", "budgeting", "accounting", "financial", "public finance", "audit"]),
]

# Turn the raw tuples into friendly dictionaries (easier to use everywhere else).
COMPETENCIES = [
    {"id": i, "name": n, "group": g, "description": d, "keywords": kw}
    for (i, n, g, d, kw) in _RAW
]
COMP_IDS = [c["id"] for c in COMPETENCIES]          # ['survey_design', 'stat_inference', ...]
COMP_INDEX = {cid: k for k, cid in enumerate(COMP_IDS)}  # {'survey_design': 0, ...}
GROUPS = ["Domain", "Functional", "Behavioural"]


# --------------------------------------------------------------------------
# PART 2: the roles.
# For every role: seniority (1 junior ... 5 senior) and a dictionary
#   {competency: required level 1-5}.
# Any competency NOT listed is "nice to have": required level = 1,
# and it gets a low importance weight (0.3) instead of 1.0.
# --------------------------------------------------------------------------
ROLES = [
    {"id": "investigator", "name": "Statistical Investigator", "seniority": 1, "needs": {
        "survey_design": 3, "spreadsheets": 3, "communication": 3, "ethics_integrity": 3,
        "official_quality": 2, "stat_inference": 2, "productivity_learning": 2,
        "quant_foundations": 2, "data_wrangling": 2, "report_writing": 2}},
    {"id": "field_supervisor", "name": "Field Supervisor", "seniority": 2, "needs": {
        "survey_design": 4, "official_quality": 3, "leadership": 3, "communication": 3,
        "spreadsheets": 3, "stat_inference": 3, "project_mgmt": 2, "ethics_integrity": 3,
        "data_wrangling": 2, "report_writing": 3, "critical_thinking": 3}},
    {"id": "stat_officer", "name": "Statistical Officer", "seniority": 3, "needs": {
        "stat_inference": 4, "survey_design": 4, "spreadsheets": 4, "data_wrangling": 3,
        "official_quality": 3, "data_viz": 3, "report_writing": 4, "python_analysis": 2,
        "sql_db": 2, "regression": 3, "macro_indicators": 3, "communication": 3,
        "critical_thinking": 3, "ethics_integrity": 3, "policy_analysis": 2,
        "time_series": 2, "quant_foundations": 3}},
    {"id": "senior_officer", "name": "Senior Statistical Officer", "seniority": 4, "needs": {
        "stat_inference": 4, "regression": 4, "time_series": 3, "survey_design": 4,
        "official_quality": 4, "report_writing": 4, "project_mgmt": 3, "leadership": 3,
        "data_viz": 4, "python_analysis": 3, "r_analysis": 3, "sql_db": 3,
        "policy_analysis": 3, "macro_indicators": 4, "research_methods": 3,
        "communication": 4, "critical_thinking": 4, "data_governance": 3,
        "ethics_integrity": 4, "quant_foundations": 3, "change_innovation": 2, "spreadsheets": 4}},
    {"id": "data_analyst", "name": "Data Analyst", "seniority": 2, "needs": {
        "python_analysis": 4, "sql_db": 4, "data_wrangling": 4, "data_viz": 4,
        "stat_inference": 4, "machine_learning": 3, "spreadsheets": 4,
        "data_storytelling": 4, "regression": 3, "r_analysis": 3, "time_series": 3,
        "critical_thinking": 3, "communication": 3, "big_data_cloud": 2,
        "quant_foundations": 3, "report_writing": 3}},
    {"id": "survey_methodologist", "name": "Survey Methodologist", "seniority": 4, "needs": {
        "survey_design": 5, "stat_inference": 5, "regression": 4, "research_methods": 4,
        "r_analysis": 4, "python_analysis": 3, "official_quality": 4,
        "quant_foundations": 4, "time_series": 3, "data_wrangling": 3,
        "report_writing": 4, "critical_thinking": 4, "data_governance": 3,
        "machine_learning": 2, "text_nlp": 2}},
    {"id": "data_engineer", "name": "Statistical IT / Data Engineer", "seniority": 3, "needs": {
        "sql_db": 5, "big_data_cloud": 4, "python_analysis": 4, "data_wrangling": 4,
        "data_governance": 4, "project_mgmt": 3, "data_viz": 2, "machine_learning": 2,
        "communication": 3, "critical_thinking": 3, "change_innovation": 3,
        "deep_learning_ai": 2}},
    {"id": "national_accounts", "name": "National Accounts Officer", "seniority": 3, "needs": {
        "macro_indicators": 5, "time_series": 4, "stat_inference": 4, "regression": 4,
        "spreadsheets": 5, "public_finance": 4, "official_quality": 4,
        "report_writing": 4, "policy_analysis": 4, "data_wrangling": 3,
        "data_viz": 3, "quant_foundations": 3, "critical_thinking": 4}},
    {"id": "gis_analyst", "name": "Geospatial (GIS) Analyst", "seniority": 3, "needs": {
        "geospatial": 5, "data_viz": 4, "python_analysis": 3, "sql_db": 3,
        "data_wrangling": 3, "stat_inference": 3, "official_quality": 3,
        "survey_design": 2, "machine_learning": 2, "report_writing": 3,
        "data_storytelling": 3, "big_data_cloud": 2, "communication": 3}},
    {"id": "dissemination", "name": "Dissemination & Communications Officer", "seniority": 3, "needs": {
        "data_viz": 4, "data_storytelling": 5, "report_writing": 5, "communication": 5,
        "spreadsheets": 3, "stat_inference": 2, "policy_analysis": 3,
        "official_quality": 3, "change_innovation": 3, "critical_thinking": 3,
        "text_nlp": 2, "ethics_integrity": 3, "data_governance": 3}},
    {"id": "deputy_director", "name": "Deputy Director (Statistics)", "seniority": 5, "needs": {
        "leadership": 5, "project_mgmt": 4, "policy_analysis": 5, "communication": 5,
        "official_quality": 4, "change_innovation": 4, "ethics_integrity": 5,
        "public_finance": 3, "critical_thinking": 5, "stat_inference": 3,
        "survey_design": 3, "macro_indicators": 3, "report_writing": 4,
        "data_governance": 4, "data_storytelling": 4, "machine_learning": 2,
        "deep_learning_ai": 2, "research_methods": 3, "productivity_learning": 3}},
]
ROLE_IDS = [r["id"] for r in ROLES]
ROLE_INDEX = {rid: k for k, rid in enumerate(ROLE_IDS)}


def role_requirements(role_id):
    """
    Return two lists (one number per competency, in COMP_IDS order):
      required[k]   = level this role must reach (1..5)
      importance[k] = 1.0 for core skills of the role, 0.3 for nice-to-have
    """
    role = ROLES[ROLE_INDEX[role_id]]
    required = [role["needs"].get(cid, 1) for cid in COMP_IDS]
    importance = [1.0 if cid in role["needs"] else 0.3 for cid in COMP_IDS]
    return required, importance


# Words for each proficiency level (shown in the dashboard)
LEVEL_NAMES = {1: "Aware", 2: "Basic", 3: "Working", 4: "Advanced", 5: "Expert"}

if __name__ == "__main__":
    print(f"{len(COMPETENCIES)} competencies, {len(ROLES)} roles")
