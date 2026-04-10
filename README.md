# AI Resume Reviewer Web App
A Streamlit-based web application that analyzes the compatibility between a candidate's resume (PDF) and a job description using AI. Built with LangChain as the orchestration framework and Google Gemini as the LLM, the app extracts text from uploaded PDF resumes, runs it through a structured LangChain LCEL pipeline, and returns a detailed match report — including a compatibility percentage, matched and missing skills, candidate strengths, improvement areas, and actionable recommendations.

**This application delivers value for two key users**: HR professionals, who can use it to speed up the CV screening process by instantly identifying the most relevant candidates, and job seekers, who can use it to better tailor their resume to a specific job description before applying — improving their chances of landing an interview.

Key Features:

- PDF Resume Upload — Automatically extracts and reads text from uploaded PDF resumes
- Match Percentage Score — Provides a 0–100% compatibility score between the resume and job description
- Verdict Label — Classifies the match level from Highly Compatible to Not Compatible
- Skill Matching — Identifies skills the candidate already possesses that align with the job requirements
- Skill Gap Detection — Highlights required skills that are missing from the resume
- Candidate Strengths — Lists key strengths relevant to the applied position
- Improvement Areas — Points out areas the candidate should develop to better fit the role
- Actionable Recommendations — Delivers concrete suggestions for both candidates and recruiters
- AI-Powered Analysis — Leverages LangChain LCEL pipeline with Google Gemini for fast, structured, and reliable output

## Installation
Clone this repository:
```bash
git clone https://github.com/ramadhaykp12/ai_resume_reviewer_web_app.git
cd ai_resume_reviewer_web_app
```

Install all required dependencies:
```bash
pip install -r requirements.txt
```
Running the program
```bash
streamlit run app.py
```


