# resume_reviewer

The aim of this project is to build a **role-focused AI resume analyzer** to help our Design Team triage 100+ applications per posting. Given a target role and a resume PDF, the app extracts text, highlights relevant keywords, surfaces strengths/areas to improve, and estimates a **match score**.

## Resume Data
Resumes are provided as user-uploaded PDFs. Text is extracted in-memory for analysis; no external data sources are used or stored.

## Role Templates
The analyzer compares resumes to editable templates in `ROLE_TEMPLATES`. Out of the box it supports:
- Machine Learning
- Computer Vision
- Firmware / Embedded
- Software (Backend)
- Control Systems / Robotics
- Mechanical

## Application
[▶ Live App](https://resumereviewerforrobosoccer.streamlit.app/) 
