# CNTA Knowledge Engineering RAG Design

**Date:** 2026-03-20

**Goal:** Reframe the existing CNTA RAG module from a document upload and keyword retrieval utility into a knowledge-engineering-oriented scientific analysis system centered on process-morphology-performance knowledge units and relation-enhanced retrieval.

## 1. Background

The current project already contains a usable knowledge-base foundation:

- SQLite-backed document, chunk, relation, and task-profile tables in [backend/core/knowledge_base.py](/D:/CNTDATA/CNTA_ML_Project/backend/core/knowledge_base.py)
- A compatibility-facing retriever in [backend/core/knowledge_rag.py](/D:/CNTDATA/CNTA_ML_Project/backend/core/knowledge_rag.py)
- RAG APIs in [backend/main.py](/D:/CNTDATA/CNTA_ML_Project/backend/main.py)
- Retrieval evaluation scripts and labeled query sets under [data/eval](/D:/CNTDATA/CNTA_ML_Project/data/eval)

However, the current primary retrieval path is still dominated by lexical matching and simple rule-based relation extraction. This is usable for demos, but it does not yet support the intended paper narrative of knowledge modeling and CNT-domain scientific analysis.

## 2. Target Positioning

The system should be positioned as:

`A knowledge-enhanced scientific analysis system for carbon nanotube arrays based on process-morphology-performance knowledge units and relation-enhanced retrieval.`

In this positioning:

- embedding retrieval is an enabling component, not the sole contribution
- the main contribution is the construction of CNT-oriented knowledge units and their semantic relations
- retrieval quality matters, but it mainly serves scientific analysis tasks

## 3. Core Scientific Tasks

The upgraded system should support three main scientific tasks:

1. Process analysis
   Goal: Given process conditions, retrieve morphology changes, mechanism interpretations, and performance implications.

2. Morphology interpretation
   Goal: Given observed SEM morphology, retrieve likely causes, growth mechanisms, and supporting literature.

3. Performance-oriented analysis
   Goal: Given a performance target or observed property, retrieve related morphology factors and process drivers.

## 4. Knowledge Unit Definition

A knowledge unit is defined as the minimum retrievable semantic fact for CNT scientific analysis. Each unit should contain:

- source metadata
- core entities
- relation type
- effect direction
- condition scope
- evidence text

### 4.1 Base Fields

- `unit_id`
- `doc_id`
- `source_type`
- `theme`
- `task_tag`
- `unit_text`
- `unit_summary`
- `keywords`
- `year`
- `is_core`
- `confidence`

### 4.2 Entity Types

- `Process`
  Examples: `growth_temp`, `growth_time`, `anneal_temp`, `anneal_time`, `ar_flow`, `h2_flow`, `c2h4_flow`, `fe_thickness`, `al2o3_thickness`

- `Morphology`
  Examples: `alignment`, `density`, `apparent_diameter`, `curvature`, `waviness`, `height`

- `Performance`
  Examples: `conductivity`, `resistivity`, `tensile_strength`, `modulus`

- `Mechanism`
  Examples: `catalyst_deactivation`, `boundary_layer_effect`, `diffusion_limitation`, `catalyst_agglomeration`, `carbon_supply_imbalance`, `stress_induced_bending`

- `Evidence`
  Examples: literature passages, experimental notes, image-analysis conclusions

## 5. Relation Schema

The main relation types should be constrained to a small, paper-friendly set:

- `process_to_morphology`
- `morphology_to_performance`
- `process_to_performance`
- `process_to_mechanism`
- `mechanism_to_morphology`
- `entity_supported_by_evidence`

The first five are the thesis-facing core relations. The last one is mainly for retrieval support and UI traceability.

Each relation record should minimally keep:

- `source_entity_type`
- `source_entity_name`
- `target_entity_type`
- `target_entity_name`
- `relation_type`
- `effect_direction`
- `condition_scope`
- `mechanism_terms`
- `evidence_text`
- `confidence`

## 6. Retrieval Architecture

The target retrieval stack is:

1. Embedding-based semantic recall
   Retrieve top-N knowledge units by vector similarity.

2. Relation enhancement
   Increase scores for candidates whose entity types, relation types, and task tags align with the query intent.

3. Light reranking
   Re-rank top-20 candidates into top-5 using a lightweight cross-encoder or equivalent pairwise scoring component.

4. Knowledge aggregation
   Assemble returned results into:
   - evidence passages
   - relation chains
   - task-oriented summaries

This design preserves the current API surface while changing the ranking logic from text-first to knowledge-unit-first retrieval.

## 7. Scientific Emphasis

The system should prioritize the following relation chains:

- `Process -> Morphology`
- `Morphology -> Performance`
- `Process -> Mechanism -> Morphology`

This directly aligns the system with CNT domain reasoning and differentiates it from a generic RAG question-answering system.

## 8. Evaluation Strategy

Retrieval should be evaluated, but only as one part of the system story.

The evaluation should include:

- retrieval effectiveness
  - Recall@k
  - MRR@k
  - nDCG@k
- relation construction quality
  - relation coverage
  - relation validity by manual inspection
- task support effectiveness
  - whether the returned chain supports process analysis, morphology interpretation, and performance explanation

## 9. Implementation Principles

- keep the current SQLite-centered architecture
- extend existing tables instead of introducing a parallel architecture unless necessary
- preserve current FastAPI endpoints and front-end integration
- make embedding retrieval the default recall path
- keep BM25 as a fallback and baseline for evaluation
- strengthen performance knowledge first with conductivity, tensile strength, and modulus

## 10. Expected Outputs

After implementation, the project should provide:

- a CNT-oriented knowledge unit schema
- enhanced relation extraction for process-morphology-performance analysis
- embedding-based primary retrieval with reranking
- a reusable evaluation set and experiment tables
- a thesis-ready system narrative centered on domain knowledge engineering rather than generic RAG
