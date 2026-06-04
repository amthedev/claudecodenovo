"""Testes do bote_engine — sanitização/validação de roteiro WhatsApp, parsing de
plano e JSON. Roda isolado (não depende de litellm/FastAPI)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_app import bote_engine as e


def test_sanitize_keeps_clean_script():
    script = "Ana: Quem fez isso???\nBia: Calma\nFOTO: PRINT DA CONVERSA\nAna: vi tudo"
    clean, warnings = e.sanitize_whatsapp_script(script)
    assert "Ana: Quem fez isso???" in clean
    assert "FOTO: PRINT DA CONVERSA" in clean
    assert warnings == []


def test_sanitize_removes_stage_directions():
    script = "Ana: oi\nAna: *abre a porta*\nBia: e ai"
    clean, warnings = e.sanitize_whatsapp_script(script)
    assert "*abre a porta*" not in clean
    assert any("acao" in w.lower() or "formato" in w.lower() for w in warnings)


def test_sanitize_removes_internal_markers_and_audio():
    script = "Ana: oi\n[Não_think interno]\nBia: mandou um audio aqui\nAna: tchau"
    clean, warnings = e.sanitize_whatsapp_script(script)
    assert "Não_think" not in clean
    assert "audio" not in clean.lower()


def test_sanitize_converts_inline_photo_to_foto_line():
    script = "Ana: olha isso [Imagem de um print]\nBia: nossa"
    clean, _ = e.sanitize_whatsapp_script(script)
    assert "FOTO:" in clean
    # a foto não pode ficar dentro da fala
    assert "[Imagem" not in clean


def test_validate_flags_photo_inside_message():
    # mensagem com palavra de foto embutida deve ser pega pela validação
    issues = e.validate_whatsapp_script("Ana: manda [foto disso] agora")
    assert any("foto" in i.lower() for i in issues)


def test_extract_json_handles_code_fence():
    raw = '```json\n{"titulo": "Teste", "partes": []}\n```'
    data = e.extract_json(raw)
    assert data["titulo"] == "Teste"


def test_extract_json_handles_surrounding_text():
    raw = 'Aqui está:\n{"a": 1}\nfim'
    assert e.extract_json(raw)["a"] == 1


def test_normalize_plan_fills_missing_parts():
    plan = e.normalize_plan({"titulo": "X", "partes": [{"numero": 1, "titulo": "P1"}]}, 3)
    assert len(plan["partes"]) == 3
    numeros = sorted(p["numero"] for p in plan["partes"])
    assert numeros == [1, 2, 3]


def test_normalize_plan_infers_characters():
    plan = e.normalize_plan(
        {"titulo": "X", "partes": [{"numero": 1, "conversa_principal": "Ana e Bia"}]}, 1
    )
    nomes = {p["nome"].lower() for p in plan["personagens"]}
    assert "ana" in nomes and "bia" in nomes


def test_conversation_quality_flags_short_script():
    issues = e.conversation_quality_issues("Ana: oi\nBia: oi")
    assert any("curta" in i.lower() or "mensagens" in i.lower() for i in issues)


def test_emoji_quality_respects_no_emoji_level():
    cfg = e.AppConfig(emoji_level="Sem emojis")
    issues = e.emoji_quality_issues("Ana: oi 😀", cfg)
    assert issues  # deve reclamar do emoji


def test_polish_capitalizes_message_start():
    cfg = e.AppConfig(emoji_level="Sem emojis")
    polished = e.polish_whatsapp_script("ana: quem foi", cfg)
    assert polished.startswith("ana: Quem foi") or "Quem foi" in polished


def test_config_from_dict_defaults():
    cfg = e.config_from_dict({"theme": "abc"})
    assert cfg.theme == "abc"
    assert cfg.size_key == "Médio"
    assert cfg.target_parts == 5
    assert cfg.model == e.CLAUDE_DEFAULT_MODEL


def test_parse_part_result_labeled_format():
    raw = "RESUMO DA PARTE:\nbriga\n\nCONVERSA:\nAna: Quem foi???\nBia: nao fui eu"
    result = e.parse_part_result(1, raw)
    assert result.numero == 1
    assert "Ana: Quem foi???" in result.roteiro


def test_extract_requested_part_count():
    assert e.extract_requested_part_count("quero com 4 partes") == 4
    assert e.extract_requested_part_count("colocar parte 6 aqui") == 6
    assert e.extract_requested_part_count("quero cinco partes") == 5
    assert e.extract_requested_part_count("nada aqui", 5) == 5


def test_dedupe_looping_lines_cuts_loop():
    """Trava de loop por código: o modelo repete a mesma fala muitas vezes →
    dedupe corta, preservando a parte coerente do início."""
    loop = (
        "Ana: oi tudo bem com voce hoje?\n"
        "Bia: tudo e voce?\n"
        "Ana: Me manda a foto de novo, por favor\n"
        "Bia: Aqui esta a foto\n"
        "Ana: Me manda a foto de novo, por favor\n"
        "Bia: Aqui esta a foto\n"
        "Ana: Me manda a foto de novo, por favor\n"
        "Bia: Aqui esta a foto\n"
        "Ana: Me manda a foto de novo, por favor\n"
        "Bia: Aqui esta a foto"
    )
    out = e.dedupe_looping_lines(loop)
    lines = [l for l in out.split("\n") if l.strip()]
    # as 2 primeiras (únicas) + 1 de cada fala repetida = bem menos que 10
    assert len(lines) < 6
    assert "oi tudo bem" in out  # início preservado


def test_dedupe_keeps_short_repeats():
    """Falas curtas comuns (sim/kkkk) podem repetir — não somem todas."""
    out = e.dedupe_looping_lines("Ana: sim\nBia: kkkk\nAna: sim")
    lines = [l for l in out.split("\n") if l.strip()]
    assert len(lines) == 3  # "sim" tolera 2
