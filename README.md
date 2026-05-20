# Context Noise in Legal RAG Systems

This repository contains the experimental framework and source code implemented for the bachelor's thesis:

> **The Impact of Context Noise on Answer Quality in Legal RAG Systems**

The project investigates how different types and amounts of context noise affect the performance of Retrieval-Augmented Generation (RAG) systems in the legal domain.

---

## Overview

Retrieval-Augmented Generation (RAG) systems improve large language model responses by incorporating external document context. However, retrieved context may contain irrelevant, semantically misleading, or contradictory information, referred to in this work as **context noise**.

This project implements a controlled experimental RAG environment designed to isolate the impact of context noise from retrieval errors. Instead of relying on automatic retrieval during experiments, relevant document fragments are preselected and noise is injected in a controlled manner.

The experiments evaluate how:

- different Signal-to-Noise Ratio (SNR) levels,
- different noise types,
- and different legal question categories

affect answer quality and context utilization.

---

## Dataset

The experiments use the following dataset:

- CUAD (Contract Understanding Atticus Dataset)

CUAD contains annotated commercial legal contracts and clause categories designed for legal NLP tasks.

Dataset repository:
https://github.com/TheAtticusProject/cuad

---

## Noise Types

The following context noise categories are implemented:

### Random Noise

Semantically unrelated legal fragments selected from different contracts and categories.

### Topical Noise

Semantically related fragments from the same legal category that do not answer the target question.

### Adversarial Noise

Semantically similar fragments containing misleading, conflicting, or contradictory information.

---

## Experimental Design

The experimental pipeline consists of:

1. Document preprocessing and chunking
2. Relevant fragment identification
3. Noise pool construction
4. Context generation with configurable SNR
5. Answer generation using `gpt-4o-mini`
6. Evaluation using the RAGAS framework

Automatic retrieval is intentionally excluded during the main experiments in order to isolate the effect of context noise from retrieval quality.

---

## Technologies

- Python
- OpenAI API
- ChromaDB
- RAGAS
- tiktoken
- pandas
- matplotlib

---

## Repository Structure

```text
├── experiment/         # Experiment execution
├── results/            # Experiment outputs and evaluation metrics
├── src/                # Core experimental framework
├── helpers/            # Helper methods
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Patriciakat/Bachelor-thesis-RAG
cd Bachelor-thesis-RAG
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

---

## Evaluation

Generated responses are evaluated using the following RAGAS metrics:

- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

The experiments analyze how these metrics change under varying context noise conditions.

---

## Thesis

This repository accompanies the bachelor's thesis submitted at:

**Vilnius University**
Faculty of Mathematics and Informatics
Software Engineering Study Programme
