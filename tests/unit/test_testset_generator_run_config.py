import sys
import types
from dataclasses import dataclass


class _FakeEncoding:
    def encode(self, text: str):  # pragma: no cover - simple stub
        return [0] * len(text)

    def decode(self, tokens):  # pragma: no cover - simple stub
        return ""


fake_tiktoken = types.ModuleType("tiktoken")
fake_tiktoken.get_encoding = lambda _name: _FakeEncoding()
fake_core = types.ModuleType("tiktoken.core")
fake_core.Encoding = _FakeEncoding
fake_tiktoken.core = fake_core
sys.modules["tiktoken"] = fake_tiktoken
sys.modules["tiktoken.core"] = fake_core

from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.dataset_schema import SingleTurnSample


class DummyLLM:
    def __init__(self):
        self.run_config = RunConfig(max_workers=3)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config


class DummyEmbeddings:
    pass


class DummyScenario:
    pass


@dataclass
class DummySynthesizer:
    name: str = "dummy"

    async def generate_scenarios(
        self,
        n: int,
        knowledge_graph,
        persona_list,
        callbacks=None,
    ):
        return [DummyScenario() for _ in range(n)]

    async def generate_sample(self, scenario, callbacks=None):
        return SingleTurnSample(user_input="question", reference="answer")


def test_generator_reuses_llm_run_config(monkeypatch):
    from ragas.testset.synthesizers import generate as generate_module
    from ragas.executor import Executor as RealExecutor

    class RecordingExecutor(RealExecutor):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            RecordingExecutor.instances.append(self)

        @classmethod
        def reset(cls):
            cls.instances = []

    RecordingExecutor.reset()
    monkeypatch.setattr(generate_module, "Executor", RecordingExecutor)

    generator = TestsetGenerator(
        llm=DummyLLM(),
        embedding_model=DummyEmbeddings(),
        knowledge_graph=KnowledgeGraph(),
        persona_list=[Persona(name="Analyst", role_description="Reviews docs")],
    )

    testset = generator.generate(
        testset_size=1,
        query_distribution=[(DummySynthesizer(), 1.0)],
    )

    assert len(testset.samples) == 1
    assert len(RecordingExecutor.instances) == 2
    for executor in RecordingExecutor.instances:
        assert executor.run_config is generator.llm.run_config
