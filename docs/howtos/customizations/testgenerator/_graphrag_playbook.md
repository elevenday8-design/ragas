# GraphRAG Testset Generation Playbook

This guide walks through every hook you can use to control GraphRAG-based testset generation in Ragas. It follows the runtime order inside `TestsetGenerator.generate` so you can decide where to plug in custom logic for your own knowledge base.【F:src/ragas/testset/synthesizers/generate.py†L289-L499】

## 1. Build the knowledge graph you want

1. **Start from domain documents.** When you call `TestsetGenerator.generate_with_langchain_docs`, each `Document` is wrapped in a `Node` with your original text and metadata before any graph logic runs.【F:src/ragas/testset/synthesizers/generate.py†L173-L196】 If you already have preprocessed nodes, instantiate `TestsetGenerator` directly with a `KnowledgeGraph` that contains your curated `Node` and `Relationship` objects.【F:src/ragas/testset/graph.py†L23-L181】
2. **Apply transforms that expose the right structure.** The default pipeline combines extractors, filters, splitters, and relationship builders chosen from document length statistics.【F:src/ragas/testset/transforms/default.py†L31-L164】 Use this as a starting point and then:
   - Clone it and tweak parameters (e.g., change thresholds, add embeddings) or
   - Assemble your own `Transforms` list/`Parallel` blocks and call `apply_transforms` yourself to populate custom properties like business-specific entities or hierarchies.【F:src/ragas/testset/transforms/engine.py†L16-L89】
3. **Persist for reuse.** `KnowledgeGraph.save` serializes nodes and relationships so you can inspect or diff changes before generating samples.【F:src/ragas/testset/graph.py†L145-L199】

### Example: custom relationship pass
```python
from ragas.testset.transforms import (
    Parallel,
    apply_transforms,
    SummaryExtractor,
    EmbeddingExtractor,
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)

transforms = [
    SummaryExtractor(llm=kg_llm, filter_nodes=lambda node: node.type.value == "document"),
    Parallel(
        EmbeddingExtractor(
            embedding_model=kg_embeddings,
            property_name="summary_embedding",
            embed_property_name="summary",
        ),
        OverlapScoreBuilder(
            property_name="business_entities",
            new_property_name="entity_overlap",
            threshold=0.05,
        ),
    ),
    CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.6,
    ),
]
apply_transforms(knowledge_graph, transforms)
```

## 2. Shape the personas driving the prompts

- Call `generate_personas_from_kg` to produce role cards from summary clusters, or pass an explicit `persona_list` when instantiating `TestsetGenerator` to lock in hand-written personas.【F:src/ragas/testset/persona.py†L62-L151】【F:src/ragas/testset/synthesizers/generate.py†L289-L397】
- Override `filter_fn` to focus persona generation on the slices of the graph that matter for your domain (e.g., only policy documents).【F:src/ragas/testset/persona.py†L62-L103】
- Swap in a custom `PersonaGenerationPrompt` (or a completely different `PydanticPrompt`) when you need more fields than `name` and `role_description`.

## 3. Decide which synthesizers run and how often

1. **Baseline options.** The default query distribution bundles one single-hop and two multi-hop synthesizers, filtering out any that cannot find candidates on the current graph.【F:src/ragas/testset/synthesizers/__init__.py†L21-L51】
2. **Tuning weights.** Pass your own list of `(synthesizer, probability)` tuples into `generate` to emphasize or suppress particular query shapes.【F:src/ragas/testset/synthesizers/generate.py†L351-L419】 Probabilities are normalized internally via `calculate_split_values`, so you can use any positive weights.
3. **Custom synthesizers.** Subclass `SingleHopQuerySynthesizer` or `MultiHopQuerySynthesizer` to change scenario discovery or prompting:
   - Override `get_node_clusters`/`_generate_scenarios` to work with bespoke graph properties such as departmental tags, escalation paths, or multi-level ownership.【F:src/ragas/testset/synthesizers/single_hop/specific.py†L41-L117】【F:src/ragas/testset/synthesizers/multi_hop/abstract.py†L31-L138】
   - Reuse `prepare_combinations`, `sample_combinations`, and `make_contexts` helpers to keep sampling balanced, or replace them entirely for deterministic coverage.【F:src/ragas/testset/synthesizers/multi_hop/base.py†L50-L186】【F:src/ragas/testset/synthesizers/single_hop/base.py†L46-L137】
   - Swap `generate_query_reference_prompt` with your own `PydanticPrompt` instance to control the final wording sent to the LLM.【F:src/ragas/testset/synthesizers/multi_hop/base.py†L50-L176】

### Example: bias abstract multi-hop towards compliance trails
```python
from dataclasses import dataclass
from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer

@dataclass
class ComplianceTrailSynth(MultiHopAbstractQuerySynthesizer):
    relation_property: str = "entity_overlap"
    abstract_property_name: str = "compliance_topics"

    def get_node_clusters(self, knowledge_graph, n=1):
        clusters = super().get_node_clusters(knowledge_graph, n)
        return [
            {node for node in cluster if node.get_property("region") == "EU"}
            for cluster in clusters
        ]
```

After defining the class, include it in your distribution:
```python
from ragas.testset.synthesizers import default_query_distribution

base = default_query_distribution(llm, knowledge_graph)
custom_distribution = [(ComplianceTrailSynth(llm=llm), 0.5), *base]
```

## 4. Orchestrate generation and runtime controls

- Call `generate` directly when your `KnowledgeGraph` and personas are prepared; otherwise use the `generate_with_*_docs` helpers for end-to-end ingestion.【F:src/ragas/testset/synthesizers/generate.py†L101-L287】
- Provide a `RunConfig` to tune retry behaviour, timeouts, worker counts, or random seeds across both scenario and sample stages. When you do not pass one explicitly, the generator now reuses the LLM's `run_config`, so setting `llm.run_config.max_workers` (or calling `set_run_config`) is enough to throttle concurrent OpenAI calls and avoid rate-limit retries.【F:src/ragas/run_config.py†L16-L90】【F:src/ragas/testset/synthesizers/generate.py†L349-L461】
- Use `batch_size` to bound simultaneous LLM calls when your provider enforces rate limits, and register callbacks (including a `CostCallbackHandler`) to capture intermediate telemetry.【F:src/ragas/testset/synthesizers/generate.py†L351-L484】
- Set `return_executor=True` to receive an `Executor` instance that you can cancel or poll manually—helpful for long multi-hop runs or staged QA workflows.【F:src/ragas/testset/synthesizers/generate.py†L468-L476】

## 5. Validate outputs and iterate

- Every generated sample is returned as a `TestsetSample`, which stores the raw `SingleTurnSample` plus the synthesizer that created it. You can diff distributions or feed them into downstream analytics before exporting.【F:src/ragas/testset/synthesizers/generate.py†L480-L499】
- Because `track` records generation metadata, you can correlate runs with instrumentation in your analytics stack when diagnosing prompt or KG changes.【F:src/ragas/testset/synthesizers/generate.py†L487-L499】

With these hooks you can decide exactly how personas, contexts, and prompts are composed, giving you fine-grained control over GraphRAG evaluation data tailored to your business knowledge base.
