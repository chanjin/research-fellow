from research_fellow.memory import KnowledgeMemory
from research_fellow.workspace_profiles import (
    BUILTIN_WORKSPACE_KEYS,
    delete_custom_workspace,
    get_workspace_profile,
    load_workspace_profiles,
    save_custom_workspace,
)
from research_fellow.workspace_archive import build_workspace_archive, restore_workspace_archive


def test_custom_workspace_roundtrip_preserves_builtins(tmp_path):
    config = tmp_path / "workspace_profiles.json"
    created = save_custom_workspace(
        config,
        key="manufacturing_ai",
        label="Manufacturing AI",
        purpose="Industrial AI research",
        expertise_instruction="Focus on manufacturing evidence.",
    )

    profiles = load_workspace_profiles(config)
    assert BUILTIN_WORKSPACE_KEYS.issubset(profiles)
    assert profiles[created.key].db_filename == "research_fellow_manufacturing_ai.db"
    assert get_workspace_profile(created.key, profiles).purpose == "Industrial AI research"

    assert delete_custom_workspace(config, created.key) is True
    profiles = load_workspace_profiles(config)
    assert created.key not in profiles
    assert BUILTIN_WORKSPACE_KEYS.issubset(profiles)


def test_builtin_workspace_cannot_be_deleted(tmp_path):
    config = tmp_path / "workspace_profiles.json"
    try:
        delete_custom_workspace(config, "general")
    except ValueError as error:
        assert "기본 워크스페이스" in str(error)
    else:
        raise AssertionError("Deleting a built-in workspace must fail")


def test_knowledge_memory_supports_count_and_paged_loading(tmp_path):
    memory = KnowledgeMemory(tmp_path / "research.db")
    for index in range(5):
        memory.add({
            "title": f"Card {index}", "claim": f"Knowledge claim number {index}",
            "source_kind": "external_paper", "provenance": {"source_name": "test"},
        })

    assert memory.count() == 5
    first_page = memory.all(limit=2)
    second_page = memory.all(limit=2, offset=2)
    assert len(first_page) == 2
    assert len(second_page) == 2
    assert {item["card_id"] for item in first_page}.isdisjoint({item["card_id"] for item in second_page})


def test_workspace_archive_can_be_reloaded_without_overwriting_local_db(tmp_path):
    config = tmp_path / "workspace_profiles.json"
    created = save_custom_workspace(config, key="robotics", label="Robotics", purpose="Robot research")
    database = tmp_path / created.db_filename
    database.write_bytes(b"local database snapshot")
    archive = build_workspace_archive(created, tmp_path)
    delete_custom_workspace(config, created.key)

    restored, database_restored = restore_workspace_archive(
        archive, config_path=config, data_dir=tmp_path,
    )

    assert restored.key == "robotics"
    assert database_restored is False
    assert database.read_bytes() == b"local database snapshot"
    assert "robotics" in load_workspace_profiles(config)
