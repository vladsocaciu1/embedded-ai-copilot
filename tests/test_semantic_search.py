from pathlib import Path

from app.chunker import chunk_index, unique_texts
from app.code_index import build_index
from app.models import ProjectFile
from app.semantic_search import embed_chunks, semantic_rank, semantic_scores
from tests.fake_embedder import FakeEmbedder


def _chunks(files: list[ProjectFile]):
    return chunk_index(build_index(files), files)


# S1
def test_s1_embed_chunks_dedups_by_key_in_first_occurrence_order():
    files = [
        ProjectFile(Path("a.c"), "void alpha(void)\n{\n    return;\n}\n"),
        ProjectFile(Path("b.c"), "void beta(void)\n{\n    return;\n}\n"),
    ]
    chunks = _chunks(files)
    chunks = chunks + (chunks[0],)
    fake = FakeEmbedder()

    embedded = embed_chunks(chunks, fake)

    texts = unique_texts(chunks)
    assert len(embedded.keys) < len(chunks)
    assert fake.document_calls == [list(texts.values())]
    assert embedded.keys == tuple(texts)
    assert embedded.chunks == chunks
    assert embedded.matrix.shape == (len(embedded.keys), fake.dims)
    assert embedded.matrix.dtype.name == "float32"


# S2
def test_s2_semantic_rank_uses_max_similarity_and_path_ties():
    question = "button debounce handler press"
    strong = Path("strong.c")
    weak = Path("weak.c")
    noise = "\n".join(
        f"void n{index:02d}(void)\n{{\n    int qqqqq{index:02d} = 1;\n}}\n"
        for index in range(12)
    )
    repeated = "\n".join(
        "    int button = 1;\n"
        "    int debounce = 1;\n"
        "    int handler = 1;\n"
        "    int press = 1;"
        for _ in range(30)
    )
    strong_body = f"void keeper(void)\n{{\n{repeated}\n}}\n"
    weak_body = (
        "void w01(void)\n{\n    int button = 1;\n}\n"
        "void w02(void)\n{\n    int debounce = 1;\n}\n"
        "void w03(void)\n{\n    int handler = 1;\n}\n"
        "void w04(void)\n{\n    int press = 1;\n}\n"
    )
    tie = "void shared_marker(void)\n{\n    int shared_marker = 1;\n}\n"
    files = [
        ProjectFile(strong, strong_body + noise),
        ProjectFile(weak, weak_body),
        ProjectFile(Path("a/tie.c"), tie),
        ProjectFile(Path("b/tie.c"), tie),
        ProjectFile(Path("notes.s"), "button debounce handler press\n"),
        ProjectFile(Path("quartz.c"), "void quartz_crystal(void)\n{\n    return;\n}\n"),
    ]
    chunks = _chunks(files)
    fake = FakeEmbedder()
    embedded = embed_chunks(chunks, fake)

    ranked = semantic_rank(question, embedded, fake)
    scores = semantic_scores(question, embedded, fake)

    assert ranked[0] == strong
    assert Path("notes.s") not in ranked
    assert set(ranked) == {chunk.path for chunk in chunks}
    assert scores[strong] > scores[weak]
    assert _mean(strong, embedded, fake, question) < _mean(weak, embedded, fake, question)
    assert scores[Path("a/tie.c")] == scores[Path("b/tie.c")]
    assert ranked.index(Path("a/tie.c")) < ranked.index(Path("b/tie.c"))
    assert ranked.index(strong) < ranked.index(Path("quartz.c"))


def _chunk_cosines(embedded, fake, question):
    import numpy as np

    query = np.asarray(fake.embed_query(question), dtype=np.float32).reshape(-1)
    cosines = embedded.matrix @ query
    index_of = {key: index for index, key in enumerate(embedded.keys)}
    return cosines, index_of


def _mean(path: Path, embedded, fake, question) -> float:
    cosines, index_of = _chunk_cosines(embedded, fake, question)
    rows = [chunk for chunk in embedded.chunks if chunk.path == path]
    return sum(float(cosines[index_of[chunk.key]]) for chunk in rows) / len(rows)


# S3
def test_s3_empty_chunks_skip_query_and_each_rank_queries_once():
    fake = FakeEmbedder()
    empty = embed_chunks((), fake)

    assert semantic_rank("button", empty, fake) == ()
    assert fake.query_calls == []
    assert fake.document_calls == []

    chunks = _chunks(
        [ProjectFile(Path("a.c"), "void alpha(void)\n{\n    return;\n}\n")]
    )
    embedded = embed_chunks(chunks, fake)
    assert semantic_rank("alpha", embedded, fake) == (Path("a.c"),)
    assert semantic_rank("alpha", embedded, fake) == (Path("a.c"),)
    assert fake.query_calls == ["alpha", "alpha"]
    assert len(fake.document_calls) == 1
