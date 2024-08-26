# PS-OOSC-GENAI
# Project Name: Web Scraping and Question Generation with Relevance Analysis

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

  
## Introduction
This project focuses on web scraping, automatic question generation, and relevance analysis. The goal is to scrape data and links from a target website, generate meaningful questions based on the scraped content, and evaluate which external links provide the best answers to these questions.

## Project Overview
1. *Web Scraping:* The project involves scraping data and hyperlinks (href attributes) from a specified website.
2. *Question Generation:* Using the T5 Transformer model, the project formulates 10 questions based on the scraped data.
3. *Relevance Analysis:* The relevance of each external link to the generated questions is evaluated using the ColBERT model from Jina AI. The analysis helps identify which links are most likely to answer the questions effectively.

## Features
- Scrape text and hyperlinks from a target website.
- Generate contextually relevant questions using the T5 Transformer model.
- Assess the relevance of external links to the generated questions using the ColBERT model.
- Output the best-matching links for each question.

## Installation
To run this project locally, follow these steps:

1. *Clone the repository:*
   bash
   git clone https://github.com/AaSiKu/PS-OOSC-GENAI.git
   
2. *Navigate to the project directory:*
   bash
   cd repository-name
   
3. *Create a virtual environment:*
   bash
   python -m venv venv
   
4. *Activate the virtual environment:*
   - On Windows:
     bash
     venv\Scripts\activate
     
   - On macOS/Linux:
     bash
     source venv/bin/activate
     
5. *Install the required dependencies:*
   bash
   pip install -r requirements.txt
   pip install git+https://github.com/stanford-futuredata/ColBERT.git 
   

## Usage
1. *Run the scraping script:*
   This script will scrape the data and hyperlinks from the specified website.
   bash
   python scrape.py
   

2. *Generate questions:*
   After scraping, run the question generation script to create 10 questions based on the scraped data.
   bash
   python generate_questions.py
   

3. *Perform relevance analysis:*
   Finally, use the relevance analysis script to evaluate which external links best answer the generated questions.
   bash
   python relevance_analysis.py
   

4. *Output:*
   The output will include the list of questions and the most relevant external links for each question.

## Dependencies
The project requires the following Python packages:

- requests
- beautifulsoup4
- transformers
- torch
- jina

Refer to the requirements.txt file for the complete list of dependencies.