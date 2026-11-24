from app.knowledge import MarkdownKnowledgeBase


async def test_retrieves_relevant_markdown_chunk_and_source(tmp_path) -> None:
    knowledge_path = tmp_path / "Knowledge"
    knowledge_path.mkdir()
    (knowledge_path / "mobile.md").write_text(
        "# Mobile Development\n\nPKCE protects OAuth authorization-code flows "
        "used by public mobile clients.\n\n# Unrelated\n\nKitchen lighting notes.",
        encoding="utf-8",
    )

    knowledge = MarkdownKnowledgeBase(str(knowledge_path), refresh_seconds=1)
    await knowledge.initialize()

    results = await knowledge.search("Why does our iOS OAuth flow need PKCE?")

    assert results
    assert results[0].source == "mobile.md"
    assert results[0].heading == "Mobile Development"
    assert "authorization-code" in results[0].content
    context = knowledge.format_context(results)
    assert "SOURCE: mobile.md — Mobile Development" in context


async def test_ignores_unrelated_markdown_and_non_markdown_files(tmp_path) -> None:
    knowledge_path = tmp_path / "Knowledge"
    knowledge_path.mkdir()
    (knowledge_path / "home.md").write_text(
        "# Heating\n\nThe boiler is controlled by Home Assistant.",
        encoding="utf-8",
    )
    (knowledge_path / "secret.txt").write_text(
        "oauth pkce should never be indexed",
        encoding="utf-8",
    )
    (knowledge_path / "._hidden.md").write_text(
        "oauth pkce should also never be indexed",
        encoding="utf-8",
    )

    knowledge = MarkdownKnowledgeBase(str(knowledge_path))
    await knowledge.initialize()

    assert await knowledge.search("OAuth PKCE mobile authentication") == ()
    assert knowledge.status()["files"] == 1


async def test_manual_refresh_picks_up_new_files(tmp_path) -> None:
    knowledge_path = tmp_path / "Knowledge"
    knowledge_path.mkdir()
    knowledge = MarkdownKnowledgeBase(str(knowledge_path), refresh_seconds=3_600)
    await knowledge.initialize()
    assert knowledge.status()["files"] == 0

    (knowledge_path / "ai.md").write_text(
        "# Retrieval\n\nRAG augments model prompts with retrieved source material.",
        encoding="utf-8",
    )
    await knowledge.refresh(force=True)

    assert (await knowledge.search("Explain RAG retrieval"))[0].source == "ai.md"


async def test_missing_knowledge_folder_is_safe(tmp_path) -> None:
    knowledge = MarkdownKnowledgeBase(str(tmp_path / "missing"))
    await knowledge.initialize()

    assert await knowledge.search("anything") == ()
    assert knowledge.status()["available"] is False
