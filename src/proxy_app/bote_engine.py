# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel
"""
bote_engine.py — lógica do WhatsApp Modeler (gerador de histórias virais em
formato de conversa de WhatsApp).

Portado de um app desktop (customtkinter) para funções puras reutilizáveis pelos
endpoints /bote/api/*. Sem dependência de UI, HTTP externo ou arquivos locais —
toda a chamada ao modelo acontece no bote_routes via RotatingClient.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-5"

SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Curto": {
        "parts": 3,
        "words": "800 a 1200 palavras no total (mínimo 250 palavras por parte)",
        "duration": "aprox. 4 a 6 minutos",
    },
    "Médio": {
        "parts": 5,
        "words": "1800 a 2500 palavras no total (mínimo 350 palavras por parte)",
        "duration": "aprox. 8 a 12 minutos",
    },
    "Longo": {
        "parts": 7,
        "words": "3000 a 4000 palavras no total (mínimo 400 palavras por parte)",
        "duration": "aprox. 15 a 20 minutos",
    },
    "Série": {
        "parts": 9,
        "words": "5000+ palavras no total (mínimo 500 palavras por parte)",
        "duration": "série longa em múltiplos vídeos",
    },
}

DRAMA_GUIDE = {
    "Baixo": "Drama cotidiano, realista, com tensão emocional moderada.",
    "Médio": "Conflito forte, revelações plausíveis e tensão crescente.",
    "Viral": "Começo impactante, prints, acusações, viradas fortes e final de parte tenso.",
}

EMOJI_GUIDE = {
    "Sem emojis": "Não use emojis em nenhuma mensagem.",
    "Leve": "Use pouquíssimos emojis: no máximo 1 ou 2 na parte inteira, só quando ficar muito natural.",
    "Médio": "Use poucos emojis: no máximo 3 ou 4 na parte inteira, sem repetir o mesmo emoji toda hora.",
    "Pesado": "Use emojis visíveis, mas ainda naturais: no máximo 6 ou 7 na parte inteira, sem virar spam.",
}

TONE_OPTIONS = [
    "Engraçada", "Caótica", "Tensa", "Raivosa", "Triste",
    "Romântica", "Fofoqueira", "Sombria",
]

EMOTION_OPTIONS = [
    "Alegria", "Tristeza", "Medo", "Raiva", "Nojo", "Surpresa", "Amor",
    "Gratidão", "Esperança", "Serenidade", "Admiração", "Orgulho",
    "Ansiedade", "Culpa", "Vergonha", "Inveja", "Ciúme",
]

FALLBACK_RULES = """
FORMATO FIXO PARA O BOT:
- Escreva somente mensagens no formato "Nome: Mensagem".
- Agrupe mensagens da mesma pessoa quando fizer sentido.
- Fotos devem aparecer em linha separada: FOTO: DESCRIÇÃO CURTA E REALISTA EM MAIÚSCULO
- Divisores de tempo devem aparecer em linha separada: [No dia seguinte]
- Nunca inclua áudio, narração fora da conversa, explicações, cabeçalhos ou roteiro.
- Nunca use mensagens vazias, reticências soltas ou linhas como "Nome: ...".
- O texto final deve estar pronto para colar no bot.
""".strip()

USER_CONVERSATION_RULES = """
REGRAS PRINCIPAIS DA JESSICA:
- A historia precisa ter comeco, meio e fim, mas nunca pode ficar chata, repetitiva ou cansativa.
- A conversa precisa parecer rotina brasileira real, com briga, emocao, fofoca, raiva, duvida, vergonha e identificacao.
- A Parte 1 precisa comecar SEMPRE com algo muito impactante, para a pessoa sentir que precisa continuar vendo o video.
- Nunca comece com "oi, tudo bem", "preciso te contar uma coisa" ou apresentacao lenta. Comece com conflito explodido.
- As conversas precisam fluir rapido, parecer vicio e fazer a pessoa ler so mais uma mensagem.
- As mensagens devem ser naturais, impulsivas, emocionais, imperfeitas, brasileiras e caoticas quando fizer sentido.
- Escreva em blocos por remetente quando fizer sentido, sem alternar linha por linha de forma robotica.
- Cada parte deve ter uma conversa principal. Se a Parte 1 comecou com Debora e Jana, essa parte fica so entre Debora e Jana.
- A proxima parte pode mudar a conversa principal, por exemplo Jana e Carlos, mas sem trocar de conversa no meio da mesma parte.
- Antes de gerar as partes, planeje quem conversa com quem em cada parte e mantenha essa decisao.
- Nao use cliffhanger como regra. Use tensao natural, revelacao, pergunta forte ou consequencia emocional quando fizer sentido.
- Cada personagem precisa ter personalidade, jeito de escrever, ritmo, girias, emojis e forma de brigar proprios.
- Mantenha o padrao de escrita dos personagens do inicio ao fim.
- Emojis precisam combinar com a personalidade e variar por personagem, sem virar spam.
- Nao use emojis de reacao do WhatsApp como conteudo isolado. Emoji precisa estar dentro de fala natural.
- A historia precisa gerar comentarios, rage bait, identificacao, compartilhamento, discussao e vontade de tomar partido.
- Se houver foto, use apenas quando ela realmente ajuda a historia e vira prova dentro da conversa.
- Imagens sao raras e estrategicas: so use FOTO para prova, documento, print, evidencia, flagrante, lugar decisivo ou virada importante.
- Nunca coloque FOTO so porque alguem esta triste, chorando, bravo ou porque a cena esta emocional.
- Antes de inserir FOTO, pergunte: essa imagem acrescenta prova, impacto ou informacao nova? Se nao, mantenha so conversa.
- Use texturas brasileiras reais: "vou nem me meter" e se mete, "KKKKK" nervoso, indireta religiosa, ironia seca, audio evitado, familia se intrometendo.
- A cada bloco importante, inclua um gatilho viral natural: revelacao por prova, silencio estrategico, virada de poder, print que muda tudo ou protagonista que ja sabia.
- Tipos de personagens devem ter jeito fixo: protagonista contida, mae 50+, sogra passivo-agressiva, homem frouxo, jovem que surta, aliada fofoqueira.
""".strip()

NATURAL_CHAT_RULES = """
CORRECAO DE NATURALIDADE:
- Nao escreva conversa com cara de novela, teatro, chantagem exagerada ou frase de vilao.
- Evite ameacas grandes e artificiais como "vou estragar tudo", "nao tem batizado", "voce ofendeu a familia" ou "e minha obrigacao".
- Prefira briga brasileira real: indireta, mensagem curta, ironia seca, pergunta atravessada, vergonha, silencio e resposta torta.
- Mensagens precisam ser curtas. A maioria deve ter ate 12 palavras. Quase nenhuma pode passar de 20 palavras.
- Nao explique demais numa mensagem so. Quebre em 2 ou 3 mensagens pequenas.
- Emoji e raro. Mesmo com emojis ligados, use so quando uma pessoa real usaria. Nunca coloque emoji em toda fala.
- Nao repita o mesmo emoji no fim de varias mensagens.
- Personagens adultos nao ficam usando emoji em toda frase durante briga seria.
- O conflito precisa parecer possivel na rotina do Brasil: familia, casa, grupo, batizado, vizinho, dinheiro, sogra, marido, mae, trabalho, entrega, print.
- Se a fala parecer "roteiro", reescreva como WhatsApp: menos perfeito, mais direto, mais humano.
""".strip()

PHOTO_RE = re.compile(r"^(?:\(\s*foto\s*:\s*(.+?)\s*\)?|foto\s*:\s*(.+?)\)?)$", re.IGNORECASE)
INLINE_PHOTO_RE = re.compile(
    r"[\[\(]\s*(?:imagem|foto)(?:\s+de)?\s*[:\-]?\s*(.+?)\s*[\]\)]",
    re.IGNORECASE,
)
PHOTO_WORD_RE = re.compile(r"[\[\(]\s*(?:imagem|foto)\b", re.IGNORECASE)
TIME_RE = re.compile(r"^\[[^\[\]]+\]$")
MESSAGE_RE = re.compile(r"^[^\s:\[\(][^:\n]{0,45}:\s+.+$")
FORBIDDEN_HEADING_RE = re.compile(r"^(resumo|roteiro|prompt|t[ií]tulo|thumbnail|parte\s+\d+|cena)\b", re.IGNORECASE)
VALID_TIME_WORD_RE = re.compile(
    r"\b(no|na|alguns?|algumas?|minutos?|horas?|dias?|dia|manha|manhã|tarde|noite|semana|depois|antes|seguinte|mais tarde|agora)\b",
    re.IGNORECASE,
)
STAGE_DIRECTION_RE = re.compile(
    r"^\*[^*]+\*$|^\([^)]*(suspiro|risos|pausa|silencio|silêncio|abre|entra|olha|pega|fecha|sai|volta|chave|porta|pia|cozinha)[^)]*\)$",
    re.IGNORECASE,
)
INTERNAL_MARKER_RE = re.compile(r"\[[^\]]*(?:think|analysis|schema|json|system|assistant|user|tool|internal|nao_|não_)[^\]]*\]", re.IGNORECASE)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


@dataclass
class AppConfig:
    theme: str = ""
    cta: str = ""
    negative_prompt: str = ""
    size_key: str = "Médio"
    model: str = CLAUDE_DEFAULT_MODEL
    drama: str = "Médio"
    emoji_level: str = "Leve"
    selected_tones: List[str] = field(default_factory=list)
    selected_emotions: List[str] = field(default_factory=list)

    @property
    def size_preset(self) -> Dict[str, Any]:
        return SIZE_PRESETS.get(self.size_key, SIZE_PRESETS["Médio"])

    @property
    def target_parts(self) -> int:
        return int(self.size_preset["parts"])


@dataclass
class PartResult:
    numero: int
    resumo: str
    roteiro: str
    prompts_imagem: List[Dict[str, str]] = field(default_factory=list)
    raw: str = ""
    warnings: List[str] = field(default_factory=list)


def config_from_dict(data: Dict[str, Any]) -> AppConfig:
    """Constrói AppConfig a partir do JSON enviado pelo frontend."""
    data = data or {}
    return AppConfig(
        theme=str(data.get("theme", "")).strip(),
        cta=str(data.get("cta", "")).strip(),
        negative_prompt=str(data.get("negative_prompt", "")).strip(),
        size_key=str(data.get("size_key") or "Médio"),
        model=str(data.get("model") or CLAUDE_DEFAULT_MODEL).strip(),
        drama=str(data.get("drama") or "Médio"),
        emoji_level=str(data.get("emoji_level") or "Leve"),
        selected_tones=list(data.get("selected_tones") or []),
        selected_emotions=list(data.get("selected_emotions") or []),
    )


# ── Helpers de texto ─────────────────────────────────────────────────────────

def count_emojis(text: str) -> int:
    count = len(EMOJI_RE.findall(text or ""))
    if count:
        return count
    surrogate_count = sum(1 for char in text or "" if unicodedata.category(char) == "Cs")
    return surrogate_count // 2


def has_emoji(text: str) -> bool:
    return count_emojis(text) > 0


def capitalize_message_start(message: str) -> str:
    for index, char in enumerate(message or ""):
        if char.isalpha():
            return message[:index] + char.upper() + message[index + 1:]
    return message


def emoji_targets_for_level(level: str) -> Tuple[int, int]:
    targets = {
        "Sem emojis": (0, 0),
        "Leve": (1, 2),
        "Médio": (2, 4),
        "Pesado": (3, 7),
    }
    return targets.get(level, (1, 2))


def natural_emoji_pool() -> List[str]:
    return [chr(0x1F633), chr(0x1F928), chr(0x1F62D), chr(0x1F621),
            chr(0x1F644), chr(0x1F615), chr(0x1F494)]


def add_missing_emojis(script: str, cfg: AppConfig) -> str:
    minimum, maximum = emoji_targets_for_level(cfg.emoji_level)
    if maximum <= 0:
        return EMOJI_RE.sub("", script or "")
    current = count_emojis(script or "")
    if current >= minimum:
        return script or ""
    needed = min(minimum - current, maximum - current)
    if needed <= 0:
        return script or ""
    output: List[str] = []
    added = 0
    pool = natural_emoji_pool()
    message_index = 0
    for raw_line in (script or "").splitlines():
        line = raw_line.strip()
        if added < needed and MESSAGE_RE.match(line) and not has_emoji(line):
            speaker, message = line.split(":", 1)
            clean_message = message.strip()
            words = re.findall(r"\w+", clean_message, flags=re.UNICODE)
            if 1 <= len(words) <= 14 and not clean_message.endswith(("?", "!", "...")):
                emoji = pool[message_index % len(pool)]
                line = f"{speaker.strip()}: {clean_message} {emoji}"
                added += 1
            message_index += 1
        output.append(line)
    return "\n".join(output)


def trim_emojis_for_level(script: str, cfg: AppConfig) -> str:
    maximum = {"Sem emojis": 0, "Leve": 2, "Médio": 4, "Pesado": 7}.get(cfg.emoji_level, 2)
    seen = 0

    def replace(match: "re.Match[str]") -> str:
        nonlocal seen
        seen += 1
        return match.group(0) if seen <= maximum else ""

    return EMOJI_RE.sub(replace, script or "")


def polish_whatsapp_script(script: str, cfg: AppConfig) -> str:
    output: List[str] = []
    for raw_line in (script or "").splitlines():
        line = raw_line.strip()
        if MESSAGE_RE.match(line) and not FORBIDDEN_HEADING_RE.match(line):
            speaker, message = line.split(":", 1)
            output.append(f"{speaker.strip()}: {capitalize_message_start(message.strip())}")
        else:
            output.append(line)
    polished = "\n".join(output)
    polished = trim_emojis_for_level(polished, cfg)
    return add_missing_emojis(polished, cfg)


def has_real_words(text: str) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÿ0-9]{2,}", text or ""))


def truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half].rstrip() + "\n\n[... trecho cortado ...]\n\n" + text[-half:].lstrip()


def remove_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


# ── JSON / parsing ───────────────────────────────────────────────────────────

def extract_json(raw: str) -> Dict[str, Any]:
    if not raw:
        raise ValueError("A IA retornou uma resposta vazia.")
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def is_valid_time_marker(line: str) -> bool:
    if not TIME_RE.match(line):
        return False
    if "_" in line or INTERNAL_MARKER_RE.search(line):
        return False
    return bool(VALID_TIME_WORD_RE.search(line))


def is_stage_direction_message(message: str) -> bool:
    clean = message.strip()
    lower = clean.lower()
    if "*" in clean or INTERNAL_MARKER_RE.search(clean):
        return True
    if STAGE_DIRECTION_RE.match(clean):
        return True
    if lower.startswith(("(", "[", "suspiro", "pausa", "silencio", "silêncio")):
        return True
    return False


def normalize_photo_description(description: str) -> str:
    description = re.sub(r"\s+", " ", (description or "").strip(" .;:-)"))
    return description.upper()


def split_inline_photo_from_message(message: str) -> Tuple[str, List[str]]:
    photos = [normalize_photo_description(m.group(1)) for m in INLINE_PHOTO_RE.finditer(message or "")]
    clean_message = INLINE_PHOTO_RE.sub("", message or "")
    clean_message = re.sub(r"\s{2,}", " ", clean_message).strip()
    return clean_message, [photo for photo in photos if photo]


def script_similarity(first: str, second: str) -> float:
    first_clean = re.sub(r"\s+", " ", (first or "").strip().lower())
    second_clean = re.sub(r"\s+", " ", (second or "").strip().lower())
    if not first_clean or not second_clean:
        return 0.0
    return difflib.SequenceMatcher(None, first_clean, second_clean).ratio()


def extract_requested_speakers(edit_request: str) -> List[str]:
    text = remove_accents(edit_request or "")
    patterns = [
        r"\bcomeca(?:r)?\s+([A-Za-z][\w'-]+)\s+e\s+([A-Za-z][\w'-]+)",
        r"\bcomece\s+com\s+([A-Za-z][\w'-]+)\s+e\s+([A-Za-z][\w'-]+)",
        r"\binicia(?:r)?\s+com\s+([A-Za-z][\w'-]+)\s+e\s+([A-Za-z][\w'-]+)",
        r"\bcom\s+([A-Za-z][\w'-]+)\s+e\s+([A-Za-z][\w'-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return [match.group(1).strip().capitalize(), match.group(2).strip().capitalize()]
    return []


def extract_requested_part_count(text: str, default: Optional[int] = None) -> Optional[int]:
    clean = remove_accents(text or "").lower()
    digit_patterns = [
        r"\b(?:coloque|colocar|inclua|incluir|adicione|adicionar|quero|ter)\s+(?:uma\s+)?parte\s+(\d+)\b",
        r"\b(?:com|em|para)\s+(\d+)\s+partes\b",
        r"\b(\d+)\s+partes\b",
    ]
    for pattern in digit_patterns:
        match = re.search(pattern, clean)
        if match:
            return max(1, int(match.group(1)))
    words = {"duas": 2, "tres": 3, "três": 3, "quatro": 4, "cinco": 5, "seis": 6}
    for word, value in words.items():
        if re.search(rf"\b(?:com|em|para|coloque|colocar|quero)\s+(?:uma\s+)?(?:parte\s+)?{word}\b", clean):
            return value
    return default


# ── Sanitização e validação de roteiro ──────────────────────────────────────

def sanitize_whatsapp_script(script: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    output: List[str] = []
    for line_number, raw_line in enumerate((script or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if line.startswith("```"):
            continue
        if INTERNAL_MARKER_RE.search(line):
            warnings.append(f"Linha {line_number}: removido marcador interno da IA.")
            continue
        if "áudio" in line.lower() or "audio" in line.lower():
            warnings.append(f"Linha {line_number}: removida menção a áudio.")
            continue
        lone_inline_photo = INLINE_PHOTO_RE.fullmatch(line)
        if lone_inline_photo:
            description = normalize_photo_description(lone_inline_photo.group(1))
            output.append(f"FOTO: {description}")
            warnings.append(f"Linha {line_number}: foto em colchetes/parenteses convertida para FOTO.")
            continue
        photo_match = PHOTO_RE.match(line)
        if photo_match:
            description = photo_match.group(1) or photo_match.group(2) or ""
            output.append(f"FOTO: {normalize_photo_description(description)}")
            continue
        if TIME_RE.match(line):
            if not is_valid_time_marker(line):
                warnings.append(f"Linha {line_number}: removido marcador interno ou divisor de tempo invalido.")
                continue
            output.append(line)
            continue
        if MESSAGE_RE.match(line) and not FORBIDDEN_HEADING_RE.match(line):
            speaker, message = line.split(":", 1)
            clean_message, inline_photos = split_inline_photo_from_message(message.strip())
            if inline_photos:
                warnings.append(f"Linha {line_number}: foto inline movida para linha FOTO separada.")
            if clean_message in {"...", "..", "."}:
                warnings.append(f"Linha {line_number}: removida reticencia solta em fala.")
                for photo_description in inline_photos:
                    output.append(f"FOTO: {photo_description}")
                continue
            if is_stage_direction_message(clean_message):
                warnings.append(f"Linha {line_number}: removida acao/indicacao tecnica dentro da fala.")
                for photo_description in inline_photos:
                    output.append(f"FOTO: {photo_description}")
                continue
            if clean_message:
                output.append(f"{speaker.strip()}: {clean_message}")
            for photo_description in inline_photos:
                output.append(f"FOTO: {photo_description}")
            continue
        warnings.append(f"Linha {line_number}: removida por não seguir o formato do bot.")
    while output and output[0] == "":
        output.pop(0)
    while output and output[-1] == "":
        output.pop()
    return "\n".join(output), warnings


def validate_whatsapp_script(script: str) -> List[str]:
    issues: List[str] = []
    lines = [line.strip() for line in (script or "").splitlines() if line.strip()]
    if not lines:
        return ["O roteiro está vazio."]
    for line_number, line in enumerate(lines, start=1):
        photo_match = PHOTO_RE.match(line)
        if photo_match:
            description = (photo_match.group(1) or photo_match.group(2) or "").strip()
            if description != description.upper():
                issues.append(f"Linha {line_number}: descrição da foto não está em maiúsculo.")
            continue
        if MESSAGE_RE.match(line):
            _speaker, message = line.split(":", 1)
            clean_message = message.strip()
            if not has_real_words(clean_message):
                issues.append(f"Linha {line_number}: mensagem sem texto real nao e permitida.")
                continue
            if PHOTO_WORD_RE.search(clean_message):
                issues.append(f"Linha {line_number}: foto dentro da fala; use FOTO em linha separada.")
                continue
            if clean_message in {"...", "..", "."}:
                issues.append(f"Linha {line_number}: reticência solta não é permitida.")
                continue
            if is_stage_direction_message(clean_message):
                issues.append(f"Linha {line_number}: acao/indicacao tecnica dentro da fala.")
                continue
        if TIME_RE.match(line):
            if not is_valid_time_marker(line):
                issues.append(f"Linha {line_number}: divisor de tempo invalido ou marcador interno.")
                continue
            continue
        if MESSAGE_RE.match(line) and not FORBIDDEN_HEADING_RE.match(line):
            continue
        issues.append(f"Linha {line_number}: formato inválido para o bot.")
    return issues


def conversation_quality_issues(script: str) -> List[str]:
    issues = validate_whatsapp_script(script)
    lines = [line.strip() for line in (script or "").splitlines() if line.strip()]
    message_lines = [line for line in lines if MESSAGE_RE.match(line)]
    speakers = {line.split(":", 1)[0].strip().lower() for line in message_lines}
    words = re.findall(r"\w+", script or "", flags=re.UNICODE)
    if len(message_lines) < 18:
        issues.append("A parte ficou curta demais; precisa de mais mensagens reais.")
    if len(words) < 220:
        issues.append("A parte tem pouco desenvolvimento emocional.")
    if len(speakers) < 2:
        issues.append("A conversa precisa de pelo menos duas pessoas falando.")
    if "*" in (script or ""):
        issues.append("A parte ainda tem acoes entre asteriscos.")
    if INTERNAL_MARKER_RE.search(script or ""):
        issues.append("A parte ainda tem marcador interno da IA.")
    if PHOTO_WORD_RE.search(script or ""):
        issues.append("A parte colocou foto/imagem dentro da fala; foto precisa ser linha separada como FOTO.")
    shout_lines: List[str] = []
    long_lines: List[str] = []
    very_long_lines: List[str] = []
    theatrical_hits: List[str] = []
    for line in message_lines:
        _speaker, message = line.split(":", 1)
        message_words = re.findall(r"\w+", message, flags=re.UNICODE)
        if len(message_words) > 16:
            long_lines.append(line)
        if len(message_words) > 24 or len(message) > 155:
            very_long_lines.append(line)
        lower_message = remove_accents(message).lower()
        if any(phrase in lower_message for phrase in [
            "vou estragar", "estrago o", "e minha obrigacao", "ofensa a familia",
            "nao tem batizado", "sem familia", "voce vai se arrepender",
            "eu exijo respeito", "nao vou permitir",
        ]):
            theatrical_hits.append(line)
        letters = re.findall(r"[A-Za-z]", message)
        if len(letters) >= 12:
            uppercase_letters = [c for c in letters if c.upper() == c and c.lower() != c.upper()]
            if letters and len(uppercase_letters) / len(letters) > 0.72:
                shout_lines.append(line)
    if len(long_lines) >= 4 or very_long_lines:
        issues.append("Muitas mensagens ficaram longas demais; precisa quebrar em falas curtas de WhatsApp.")
    if theatrical_hits:
        issues.append("A conversa ficou teatral/artificial demais; troque ameacas grandes por briga brasileira mais real.")
    if len(shout_lines) >= 3 or (message_lines and len(shout_lines) / len(message_lines) > 0.18):
        issues.append("A conversa ficou gritada demais em CAPS LOCK; maiusculo so em FOTO ou uma explosao pontual.")
    return issues


def emoji_quality_issues(script: str, cfg: AppConfig) -> List[str]:
    count = count_emojis(script or "")
    if cfg.emoji_level == "Sem emojis":
        return ["A opcao Sem emojis foi escolhida, mas a conversa veio com emoji."] if count else []
    maximum = {"Leve": 2, "Médio": 4, "Pesado": 7}.get(cfg.emoji_level, 2)
    if count > maximum:
        return [f"A conversa exagerou nos emojis para o nivel {cfg.emoji_level}; use no maximo {maximum} na parte inteira."]
    message_lines = [line for line in (script or "").splitlines() if MESSAGE_RE.match(line.strip())]
    lines_with_emoji = [line for line in message_lines if has_emoji(line)]
    if message_lines and len(lines_with_emoji) / len(message_lines) > 0.25:
        return ["Emojis apareceram em falas demais; deixe emojis raros e pontuais."]
    return []


def script_message_lines(script: str) -> List[str]:
    return [line.strip() for line in (script or "").splitlines()
            if MESSAGE_RE.match(line.strip()) and not PHOTO_RE.match(line.strip())]


def script_speakers(script: str) -> List[str]:
    speakers: List[str] = []
    for line in script_message_lines(script):
        speaker = line.split(":", 1)[0].strip()
        if speaker.lower() not in {item.lower() for item in speakers}:
            speakers.append(speaker)
    return speakers


def extract_main_speakers(value: str) -> List[str]:
    text = re.sub(r"\bou\b|\bgrupo\b|\bfamilia\b", ",", remove_accents(value or ""), flags=re.IGNORECASE)
    names: List[str] = []
    for chunk in re.split(r"\s+e\s+|,|/|\+|\s+com\s+", text, flags=re.IGNORECASE):
        name = chunk.strip(" .:-")
        if not name or len(name) > 35:
            continue
        if name.lower() in {"pessoa a", "pessoa b", "nome do grupo", "definir duas pessoas"}:
            continue
        if name.lower() not in {item.lower() for item in names}:
            names.append(name)
    return names


def normalized_message_text(line: str) -> str:
    if ":" in line:
        line = line.split(":", 1)[1]
    line = remove_accents(line).lower()
    line = re.sub(r"[^\w\s]+", "", line, flags=re.UNICODE)
    return re.sub(r"\s+", " ", line).strip()


def continuity_quality_issues(script: str, previous_parts: Dict[int, PartResult],
                              part_number: int, part_spec: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if part_number <= 1 or not previous_parts:
        return issues
    current_lines = script_message_lines(script)
    if not current_lines:
        return issues
    previous_part = previous_parts.get(part_number - 1)
    if previous_part:
        prev_lines = script_message_lines(previous_part.roteiro)
        prev_end = [normalized_message_text(line) for line in prev_lines[-8:]]
        current_start = [normalized_message_text(line) for line in current_lines[:10]]
        repeated_start = [line for line in current_start if line and line in prev_end]
        if repeated_start:
            issues.append("A parte atual repetiu falas do final da parte anterior; precisa continuar depois do que ja aconteceu.")
        prev_texts = {normalized_message_text(line) for line in prev_lines if normalized_message_text(line)}
        current_texts = [normalized_message_text(line) for line in current_lines if normalized_message_text(line)]
        repeated_count = sum(1 for line in current_texts if line in prev_texts)
        if repeated_count >= 3:
            issues.append("A parte atual reaproveitou falas demais de partes anteriores; parece a mesma cena de novo.")
        prev_tail = " ".join(prev_end)
        current_head = " ".join(current_start)
        if script_similarity(prev_tail, current_head) > 0.60:
            issues.append("O inicio da parte atual parece o mesmo fechamento da parte anterior.")
    expected_speakers = extract_main_speakers(str(part_spec.get("conversa_principal") or ""))
    if len(expected_speakers) >= 2:
        allowed = {name.lower() for name in expected_speakers}
        actual = script_speakers(script)
        outsiders = [name for name in actual if name.lower() not in allowed]
        if outsiders:
            issues.append("A parte trocou de conversa no meio. Remetentes fora da conversa principal: "
                          + ", ".join(outsiders[:4]) + ".")
    return issues


def edit_request_quality_issues(script: str, edit_request: str, mode: str) -> List[str]:
    issues: List[str] = []
    requested_speakers = extract_requested_speakers(edit_request)
    if requested_speakers:
        speakers = {line.split(":", 1)[0].strip().lower()
                    for line in (script or "").splitlines() if MESSAGE_RE.match(line.strip())}
        missing = [name for name in requested_speakers if name.lower() not in speakers]
        if missing:
            issues.append("O pedido mandou usar " + " e ".join(requested_speakers)
                          + ", mas a conversa gerada nao usou esses nomes como personagens principais.")
    if mode == "new" and requested_speakers:
        first_message = next((line for line in (script or "").splitlines() if MESSAGE_RE.match(line.strip())), "")
        if first_message and not any(first_message.lower().startswith(f"{name.lower()}:") for name in requested_speakers):
            issues.append("A nova versao precisa comecar ja com um dos personagens pedidos pela usuaria.")
    return issues


# ── Plano ────────────────────────────────────────────────────────────────────

def infer_characters_from_parts(parts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    names: List[str] = []
    for part in parts:
        conversa = str(part.get("conversa_principal") or "")
        for name in re.split(r"\s+e\s+|,|/|\+|\s+com\s+", conversa, flags=re.IGNORECASE):
            clean = name.strip(" .:-")
            if not clean or clean.lower() in {"grupo", "familia", "pessoa a", "pessoa b"}:
                continue
            if len(clean) > 30:
                continue
            if clean.lower() not in {existing.lower() for existing in names}:
                names.append(clean)
    return [{"nome": name, "perfil": f"Personagem principal da conversa: {name}.",
             "funcao": "definir na geracao de personagens",
             "personalidade": "manter coerente com o plano",
             "jeito_de_escrever": "definir pelo contexto da historia",
             "emojis": "usar de forma natural"} for name in names]


def normalize_plan(data: Dict[str, Any], target_parts: int) -> Dict[str, Any]:
    data = dict(data or {})
    target_parts = max(1, int(target_parts or 1))
    data.setdefault("titulo", "Historia de WhatsApp")
    data.setdefault("resumo", {})
    data.setdefault("personagens", [])
    data.setdefault("partes", [])
    data["prompts_imagem_gerais"] = []
    if isinstance(data.get("personagens"), list):
        for person in data["personagens"]:
            if isinstance(person, dict):
                person.pop("prompt_imagem", None)
    parts = data.get("partes")
    if not isinstance(parts, list) or not parts:
        data["partes"] = [{
            "numero": i, "titulo": f"Parte {i}",
            "acontecimentos": "Acontecimentos a definir.",
            "conversa_principal": "Definir duas pessoas ou um grupo fixo.",
            "gancho_inicio": "Comeco impactante." if i == 1 else "Entrada direta na conversa.",
            "objetivo": "Avancar a historia sem trocar de conversa no meio.",
        } for i in range(1, target_parts + 1)]
    else:
        normalized_parts: List[Dict[str, Any]] = []
        for index, part in enumerate(parts, start=1):
            if not isinstance(part, dict):
                continue
            part = dict(part)
            try:
                number = int(part.get("numero") or index)
            except (TypeError, ValueError):
                number = index
            part["numero"] = number
            part.setdefault("titulo", f"Parte {number}")
            part.setdefault("acontecimentos", "")
            part.setdefault("conversa_principal", "")
            part.setdefault("gancho_inicio", "")
            part.setdefault("objetivo", "")
            part.pop("cliffhanger", None)
            normalized_parts.append(part)
        normalized_parts.sort(key=lambda item: int(item.get("numero") or 0))
        existing_numbers = {int(part.get("numero") or 0) for part in normalized_parts}
        for number in range(1, target_parts + 1):
            if number in existing_numbers:
                continue
            normalized_parts.append({
                "numero": number, "titulo": f"Parte {number}",
                "acontecimentos": "Parte adicionada para completar a quantidade pedida. Desenvolver uma nova virada da historia sem repetir as partes anteriores.",
                "conversa_principal": "Definir duas pessoas ou um grupo fixo.",
                "gancho_inicio": "Entrada direta com prova, acusacao ou mensagem dificil de ignorar.",
                "objetivo": "Avancar a historia com emocao clara e conexao com o publico.",
            })
        normalized_parts.sort(key=lambda item: int(item.get("numero") or 0))
        data["partes"] = normalized_parts
    if not isinstance(data.get("personagens"), list) or not data.get("personagens"):
        data["personagens"] = infer_characters_from_parts(data.get("partes", []))
    return data


def parse_plan_title(text: str) -> str:
    lines = (text or "").splitlines()
    for index, line in enumerate(lines):
        clean = re.sub(r"[^a-z0-9]+", "", remove_accents(line).lower())
        if clean.startswith("titulo"):
            if ":" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
            for next_line in lines[index + 1:]:
                value = next_line.strip()
                if value:
                    return value
    return "Historia de WhatsApp"


def parse_plan_field(section: str, labels: List[str]) -> str:
    wanted = [re.sub(r"[^a-z0-9]+", "", remove_accents(label).lower()) for label in labels]
    for line in (section or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_clean = re.sub(r"[^a-z0-9]+", "", remove_accents(key).lower())
        if any(key_clean.startswith(label) or label.startswith(key_clean) for label in wanted if key_clean):
            return value.strip()
    return ""


def parse_plan_text(raw: str, target_parts: int) -> Dict[str, Any]:
    text = (raw or "").strip()
    normalized = remove_accents(text)
    title = parse_plan_title(text)
    characters: List[Dict[str, Any]] = []
    char_block_match = re.search(r"(?is)(?:PERFIL DOS PERSONAGENS|PERSONAGENS)\s*:?\s*(.*?)(?:ESTRUTURA DAS PARTES|PARTE\s+1\s*:)", normalized)
    if char_block_match:
        original_block = text[char_block_match.start(1):char_block_match.end(1)]
        for line in original_block.splitlines():
            line = line.strip()
            if not line:
                continue
            body = line.lstrip("-0123456789. ").strip()
            if not body or (":" in body.lower()[:12] and not body.lower().startswith(("nome", "personagem"))):
                continue
            chunks = [chunk.strip() for chunk in re.split(r";|\n", body) if chunk.strip()]
            for chunk in chunks:
                name = chunk.split(",", 1)[0].split("-", 1)[0].split(":", 1)[-1].strip()
                if name and len(name) <= 35:
                    characters.append({"nome": name, "perfil": chunk, "funcao": "",
                                       "personalidade": "", "jeito_de_escrever": "", "emojis": ""})
    parts: List[Dict[str, Any]] = []
    part_matches = list(re.finditer(r"(?im)^\s*PARTE\s+(\d+)\s*:?\s*(.*)$", normalized))
    for index, match in enumerate(part_matches):
        number = int(match.group(1))
        section_start = match.end()
        section_end = part_matches[index + 1].start() if index + 1 < len(part_matches) else len(text)
        section = text[section_start:section_end].strip()
        normalized_section = remove_accents(section)
        original_title = text[match.start(2):match.end(2)].strip()
        parts.append({
            "numero": number,
            "titulo": original_title or f"Parte {number}",
            "conversa_principal": parse_plan_field(normalized_section, ["Conversa principal", "Conversa", "Quem conversa"]),
            "gancho_inicio": parse_plan_field(normalized_section, ["Comeco/entrada", "Comeco", "Entrada impactante", "Gancho inicial", "Inicio"]),
            "objetivo": parse_plan_field(normalized_section, ["Objetivo", "Emocao", "Emocao/revelacao"]),
            "acontecimentos": parse_plan_field(normalized_section, ["Acontece", "Acontecimentos", "Resumo", "O que acontece"]) or section[:500],
        })
    if not parts:
        parts = [{"numero": i, "titulo": f"Parte {i}", "conversa_principal": "",
                  "gancho_inicio": "Comeco impactante." if i == 1 else "",
                  "acontecimentos": "", "objetivo": ""} for i in range(1, target_parts + 1)]
    return normalize_plan({"titulo": title, "resumo": {"inicio": "", "meio": "", "fim": ""},
                           "personagens": characters, "partes": parts}, target_parts)


def get_part_count(plan: Optional[Dict[str, Any]], default: int) -> int:
    if plan and isinstance(plan.get("partes"), list) and plan["partes"]:
        return len(plan["partes"])
    return default


def render_plan(plan: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"TÍTULO PROVISÓRIO\n{plan.get('titulo', '')}\n")
    resumo = plan.get("resumo", {})
    if isinstance(resumo, dict):
        lines.append("RESUMO DA TRAMA")
        lines.append(f"Início: {resumo.get('inicio', '')}")
        lines.append(f"Meio: {resumo.get('meio', '')}")
        lines.append(f"Fim: {resumo.get('fim', '')}\n")
    lines.append("PERFIL DOS PERSONAGENS")
    for person in plan.get("personagens", []):
        if not isinstance(person, dict):
            continue
        lines.append(f"- {person.get('nome', 'Personagem')}, {person.get('idade', '?')} anos: {person.get('perfil', '')}")
        if person.get("personalidade"):
            lines.append(f"  Personalidade: {person.get('personalidade')}")
        if person.get("jeito_de_escrever"):
            lines.append(f"  Jeito de escrever: {person.get('jeito_de_escrever')}")
        if person.get("emojis"):
            lines.append(f"  Emojis/gírias: {person.get('emojis')}")
    lines.append("")
    lines.append("ESTRUTURA DAS PARTES")
    for part in plan.get("partes", []):
        if not isinstance(part, dict):
            continue
        lines.append(f"Parte {part.get('numero')}: {part.get('titulo', '')}")
        lines.append(f"Conversa principal: {part.get('conversa_principal', '')}")
        lines.append(f"Começo/entrada: {part.get('gancho_inicio', '')}")
        lines.append(f"Acontece: {part.get('acontecimentos', '')}")
        lines.append(f"Objetivo: {part.get('objetivo', '')}\n")
    return "\n".join(lines).strip()


# ── Parsing de partes ────────────────────────────────────────────────────────

def parse_labeled_part_response(raw: str) -> Tuple[str, str]:
    text = (raw or "").strip()
    if not text:
        return "", ""
    resumo = ""
    script = text
    match = re.search(r"(?is)(?:RESUMO(?:\s+DA\s+PARTE)?\s*:)(.*?)(?:\n\s*(?:CONVERSA|ROTEIRO_WHATSAPP|ROTEIRO)\s*:)(.*)", text)
    if match:
        resumo = match.group(1).strip()
        script = match.group(2).strip()
    else:
        script_match = re.search(r"(?is)(?:CONVERSA|ROTEIRO_WHATSAPP|ROTEIRO)\s*:\s*(.*)", text)
        if script_match:
            script = script_match.group(1).strip()
    script = re.split(r"(?im)^\s*(?:PROMPTS?\s+DE\s+IMAGEM|PROMPT\s+DA\s+FOTO|OBSERVAÇÕES|OBSERVACOES)\s*:", script)[0].strip()
    return resumo, script


def normalize_prompts(prompts: Any) -> List[Dict[str, str]]:
    if not isinstance(prompts, list):
        return []
    normalized: List[Dict[str, str]] = []
    for index, item in enumerate(prompts, start=1):
        if isinstance(item, dict):
            scene = str(item.get("cena") or item.get("scene") or f"Cena {index}").strip()
            prompt = str(item.get("prompt") or item.get("image_prompt") or "").strip()
            part = str(item.get("parte") or item.get("part") or "").strip()
            chat_excerpt = str(item.get("trecho_conversa") or item.get("chat_excerpt") or "").strip()
            characters = item.get("personagens") or item.get("characters") or []
            visible_text = str(item.get("visible_text") or "").strip()
        else:
            scene, prompt, part, chat_excerpt, characters, visible_text = f"Cena {index}", str(item).strip(), "", "", [], ""
        if prompt:
            normalized.append({
                "cena": scene or f"Cena {index}", "parte": part, "trecho_conversa": chat_excerpt,
                "personagens": characters if isinstance(characters, list) else [],
                "prompt": prompt, "visible_text": visible_text,
            })
    return normalized


def parse_part_result(part_number: int, raw: str) -> PartResult:
    try:
        data = extract_json(raw)
        script = data.get("roteiro_whatsapp", "")
        resumo = data.get("resumo_parte", "")
        prompts = data.get("prompts_imagem") or []
    except Exception:
        resumo, script = parse_labeled_part_response(raw)
        prompts = []
    clean_script, warnings = sanitize_whatsapp_script(script)
    return PartResult(numero=part_number, resumo=resumo or f"Parte {part_number} gerada.",
                      roteiro=clean_script, prompts_imagem=normalize_prompts(prompts),
                      raw=raw, warnings=warnings)


# ── Builders de prompt ───────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return f"""
Você é um roteirista sênior de histórias virais brasileiras em formato WhatsApp.
Siga as regras abaixo com rigor.

REGRAS DO BOT:
{FALLBACK_RULES}

REGRAS DA JESSICA E EXEMPLOS:
{USER_CONVERSATION_RULES}

REGRAS MAIS IMPORTANTES DE NATURALIDADE:
{NATURAL_CHAT_RULES}

CONTRATO PRINCIPAL:
- Na etapa de plano, monte titulo, resumo, personagens e estrutura das partes sem usar cliffhanger como regra.
- Na etapa de conversa, entregue somente texto pronto para o bot.
- Nunca coloque narração, cabeçalhos, explicações, markdown ou áudio dentro da conversa.
- Fotos devem sair como FOTO: DESCRIÇÃO CURTA E REALISTA EM MAIÚSCULO.
- Divisores de tempo devem sair em linha separada no formato [No dia seguinte].
- Nunca use mensagens vazias, reticências soltas ou linhas como "Nome: ...".
- Cada parte deve manter uma conversa principal fixa. Nao troque de dupla/grupo no meio da parte.
- A primeira mensagem da Parte 1 precisa ser muito impactante.
""".strip()


def build_plan_prompt(cfg: AppConfig) -> str:
    preset = cfg.size_preset
    tone_text = ", ".join(cfg.selected_tones) if cfg.selected_tones else "Livre"
    emotion_text = ", ".join(cfg.selected_emotions) if cfg.selected_emotions else "Livre"
    return f"""
PLANO DA HISTORIA
Responda normal, em texto organizado, sem JSON e sem markdown.
Use frases claras e compactas para eu ler e aprovar antes das partes.

Tema/nicho:
{cfg.theme}

Tamanho selecionado: {cfg.size_key} ({preset['words']}, {preset['duration']})
Quantidade-alvo de partes: {cfg.target_parts}
Nível de fofoca/drama: {cfg.drama} - {DRAMA_GUIDE.get(cfg.drama, '')}
Nível de emojis: {cfg.emoji_level} - {EMOJI_GUIDE.get(cfg.emoji_level, '')}
Tons desejados: {tone_text}
Emoções que a história deve despertar: {emotion_text}
CTA desejada: {cfg.cta or 'Não informado'}
Prompt negativo:
{cfg.negative_prompt or 'Não informado'}

Crie um plano de história viral brasileira realista, com começo, meio e fim.
Estruture exatamente {cfg.target_parts} partes, sem usar cliffhanger como regra.
Nesta etapa, gere somente plano e perfis basicos. Nao gere prompts de personagens, cenas, imagem ou video.

Regras específicas da estrutura:
- As emoções selecionadas precisam aparecer como motor da trama, não apenas como tema.
- Cada parte deve definir a conversa principal com pessoas fixas.
- Dentro da mesma parte, nao troque para outra conversa.
- A Parte 1 deve comecar com uma primeira mensagem muito impactante, impossivel de ignorar.
- Explique a origem das provas e revelações. Nada pode surgir do nada.
- O título precisa sugerir antagonista + injustiça + consequência.
- A protagonista não deve vencer cedo demais. A virada precisa ser construída.

Formato da resposta:
TITULO: titulo provisório
RESUMO: começo, meio e fim em poucas linhas
PERSONAGENS: nome, função, personalidade, jeito de escrever, gírias/emojis de cada um
PARTE 1: nome da parte
Conversa principal: Pessoa A e Pessoa B, ou nome do grupo
Começo: primeira mensagem impactante
Acontecimentos: o que acontece
Objetivo: emoção/revelação que a parte precisa provocar
Repita esse formato para todas as partes.
""".strip()


def build_plan_revision_prompt(cfg: AppConfig, current_plan_text: str, edit_request: str) -> str:
    requested_parts = extract_requested_part_count(edit_request)
    part_rule = (f"A usuaria pediu {requested_parts} partes. Obedeca isso e entregue exatamente {requested_parts} partes."
                 if requested_parts else f"Mantenha exatamente a quantidade de partes selecionada: {cfg.target_parts}.")
    return f"""
REFINAR PLANO DA HISTORIA
Responda normal, em texto organizado, sem JSON e sem markdown.

Voce vai refazer apenas o plano da historia, obedecendo o pedido da usuaria.
Nao gere a conversa das partes ainda.

Pedido da usuaria para mudar o plano:
{edit_request}

Plano atual:
{truncate_text(current_plan_text, 14000)}

Regras:
- O pedido da usuaria manda mais que o plano atual.
- {part_rule}
- Preserve o que estiver bom e mude claramente o que foi pedido.
- Cada parte deve continuar com uma conversa principal fixa.
- A Parte 1 ainda precisa comecar com algo muito impactante.
- Nao use cliffhanger como regra.

Formato da resposta:
TITULO: titulo provisório
RESUMO: começo, meio e fim em poucas linhas
PERSONAGENS: nome, função, personalidade, jeito de escrever, gírias/emojis de cada um
PARTE 1: nome da parte
Conversa principal: Pessoa A e Pessoa B, ou nome do grupo
Começo: primeira mensagem impactante
Acontecimentos: o que acontece
Objetivo: emoção/revelação que a parte precisa provocar
Repita esse formato para todas as partes.
""".strip()


def build_part_prompt(cfg: AppConfig, plan: Dict[str, Any], part_number: int,
                      previous_parts: Dict[int, PartResult]) -> str:
    expected_parts = get_part_count(plan, cfg.target_parts)
    previous_context = []
    for number in sorted(previous_parts):
        if number >= part_number:
            continue
        part = previous_parts[number]
        previous_context.append(f"### PARTE {number} JÁ GERADA\nResumo: {part.resumo}\nRoteiro:\n{part.roteiro}")
    tone_text = ", ".join(cfg.selected_tones) if cfg.selected_tones else "Livre"
    emotion_text = ", ".join(cfg.selected_emotions) if cfg.selected_emotions else "Livre"
    part_spec: Dict[str, Any] = {}
    for item in plan.get("partes", []):
        if isinstance(item, dict) and int(item.get("numero", 0) or 0) == part_number:
            part_spec = item
            break
    continuity_rules = ""
    if part_number > 1:
        continuity_rules = """
REGRAS DE CONTINUIDADE ENTRE PARTES:
- Esta parte NAO e uma historia nova. E continuacao direta das partes anteriores.
- Leia o final da ultima parte aprovada e comece depois dele.
- Nao reabra como novo um conflito que ja foi decidido.
- Nao copie frases, encerramentos ou batidas emocionais da parte anterior.
- A conversa principal desta parte e obrigatoria. Quem nao estiver nela pode ser citado, mas nao pode virar remetente.
- Nao use FOTO ou novo personagem como atalho para mudar de conversa no meio.
""".strip()
    return f"""
PARTE DA CONVERSA
Responda normal, sem JSON e sem markdown.
Entregue primeiro um resumo curto e depois a conversa pronta para colar no bot.

PARTE_SOLICITADA: {part_number}
TOTAL_DE_PARTES: {expected_parts}

Plano da história:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Parte planejada agora:
{json.dumps(part_spec, ensure_ascii=False, indent=2)}

Contexto das partes anteriores:
{truncate_text(chr(10).join(previous_context) or 'Nenhuma parte anterior.', 18000)}

{continuity_rules}

Nível de fofoca/drama: {cfg.drama} - {DRAMA_GUIDE.get(cfg.drama, '')}
Nível de emojis: {cfg.emoji_level} - {EMOJI_GUIDE.get(cfg.emoji_level, '')}
Tons desejados nesta história: {tone_text}
Emoções que a parte deve provocar: {emotion_text}
Tamanho geral: {cfg.size_key} - {cfg.size_preset['words']}; esta parte deve ter no mínimo 15 mensagens e 250 palavras de diálogo.
CTA do usuário: {cfg.cta or 'Não informado'}
Prompt negativo do usuário:
{cfg.negative_prompt or 'Não informado'}

Regras absolutas para a conversa:
- Nesta etapa, gere somente a conversa da parte. Nao gere personagens, cenas, prompts de imagem ou prompts de video.
- As emoções escolhidas ({emotion_text}) precisam aparecer na conversa por meio de fala, acusação, medo, raiva, surpresa, vergonha, ciúme, ironia, silêncio quebrado ou reação impulsiva.
- Não escreva uma conversa limpa demais. Ela precisa ter cortes, mensagens curtas, reação atravessada, brasileiro real e ritmo de fofoca.
- Nao escreva falas inteiras em CAPS LOCK. Maiusculo so em FOTO ou em uma explosao pontual de 1 frase curta.
- Se for Parte 1, a primeira mensagem precisa ser muito impactante.
- Use a conversa principal definida no plano para esta parte e nao troque de conversa no meio.
- Use somente linhas "Nome: Mensagem", "FOTO: DESCRIÇÃO..." e "[Divisor de tempo]".
- Toda fala precisa comecar com letra maiuscula depois dos dois-pontos.
- Fotos NUNCA podem ficar dentro de uma mensagem. Foto/prova sempre precisa ser uma linha separada: "FOTO: DESCRICAO EM MAIUSCULO".
- Nao use asteriscos para acao. Nunca escreva "*abre a porta*", "(suspiro)", "(pausa)".
- Nao use marcadores internos como [Não_think], [think], [analysis], [sistema], [schema].
- Divisores em colchetes so podem ser tempo natural, como [No dia seguinte].
- Cada parte precisa ter no minimo 18 mensagens reais, com conflito claro e uma virada ou revelação.
- A maioria das mensagens deve ser curta, com ate 12 palavras. Se uma fala passar de 16 palavras, quebre em mais mensagens.
- Nao use frases de novela ou ameaca artificial.
- Quando emojis estiverem ligados, use alguns emojis naturais: Leve = 1 ou 2 na parte; Medio = 2 a 4; Pesado = 3 a 7. Sem emojis = zero.
- Na última parte, feche a história com começo, meio e fim bem amarrados.

Formato da resposta:
RESUMO DA PARTE:
um resumo curto

CONVERSA:
Nome: Mensagem
Nome: Mensagem
FOTO: DESCRICAO CURTA E REALISTA EM MAIUSCULO
[Algumas horas depois]
Nome: Mensagem
""".strip()


def build_rewrite_part_prompt(original_prompt: str, bad_script: str, issues: List[str]) -> str:
    return f"""
REFAZER PARTE
Refaça a parte inteira. Responda normal, sem JSON e sem markdown.

A primeira tentativa foi recusada porque ficou fraca ou com formato ruim:
{chr(10).join(f"- {issue}" for issue in issues[:12])}

Trecho ruim recebido:
{truncate_text(bad_script, 5000)}

Pedido original:
{truncate_text(original_prompt, 18000)}

Regras de correção:
- Entregue uma conversa de WhatsApp humana, brasileira, emocional e direta.
- Quebre mensagens longas. A maioria precisa ter ate 12 palavras.
- Nao use asteriscos, parenteses de acao, marcadores internos ou colchetes tecnicos.
- Fotos NUNCA podem ficar dentro da fala. Use sempre uma linha separada: FOTO: DESCRICAO EM MAIUSCULO.
- Use pelo menos 22 mensagens reais.
- Nao transforme a cena em teatro. Tem que parecer print de conversa.
- Nao use CAPS LOCK em varias falas. Maiusculo so para FOTO ou uma explosao curta e rara.

Formato da resposta:
RESUMO DA PARTE:
um resumo curto

CONVERSA:
Nome: Mensagem
Nome: Mensagem
""".strip()


def build_edit_part_prompt(original_prompt: str, current_script: str, edit_request: str, mode: str) -> str:
    mode_text = ("Refaca a mesma parte inteira, corrigindo a versao atual de acordo com o pedido da usuaria, mas sem fazer ajuste pequeno."
                 if mode == "again"
                 else "Crie uma versao nova da mesma parte, sem ficar presa a versao atual. Se a usuaria pedir outros personagens, use os personagens pedidos e ignore a dupla/grupo antigo dessa parte.")
    edit_text = edit_request or "Melhorar a parte: mais emocao, mais naturalidade, mais ritmo de WhatsApp e mais conexao com as emocoes escolhidas."
    return f"""
EDITAR PARTE
Responda normal, sem JSON e sem markdown.

{mode_text}

Pedido da usuaria para esta parte:
{edit_text}

Versao atual da parte:
{truncate_text(current_script or 'Nenhuma versao atual.', 7000)}

Pedido original da parte:
{truncate_text(original_prompt, 18000)}

Regras:
- Entregue RESUMO DA PARTE e CONVERSA.
- A conversa precisa ficar pronta para colar no bot.
- Toda fala precisa comecar com letra maiuscula depois dos dois-pontos.
- Dentro da versao final, mantenha uma unica conversa principal. Nao troque de dupla/grupo no meio.
- Obedeca o pedido de edicao acima como prioridade absoluta.
- Nao troque apenas algumas palavras. Reescreva a parte de verdade, com pelo menos 60% das falas diferentes.
- Fotos nunca ficam dentro da fala. Use sempre linha separada: FOTO: DESCRICAO EM MAIUSCULO.

Formato:
RESUMO DA PARTE:
um resumo curto

CONVERSA:
Nome: Mensagem
Nome: Mensagem
""".strip()


def build_character_prompt(plan: Dict[str, Any], parts: Dict[int, PartResult]) -> str:
    scripts = [f"### PARTE {number}\n{parts[number].roteiro}" for number in sorted(parts)]
    return f"""
PERSONAGENS DA HISTORIA
Responda normal, sem JSON e sem markdown.

Voce vai montar TODOS os personagens depois que as conversas ja foram aprovadas.
Nao gere cenas. Nao gere prompts de video. Nao reescreva as conversas.

Plano aprovado:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Conversas aprovadas:
{truncate_text(chr(10).join(scripts), 22000)}

Regras:
- Inclua todos os personagens citados ou importantes para a historia.
- Defina girias, emojis, ritmo de mensagem, nivel de grosseria/carinho e jeito de brigar de cada um.
- Os prompts de imagem devem vir em ingles.
- Texto visivel, se existir, pode ficar em portugues.

Formato da resposta:
PERSONAGEM: Nome
Idade:
Função:
Perfil:
Personalidade:
Jeito de escrever:
Emojis/gírias:
Prompt de imagem:

Repita para todos os personagens.
""".strip()


def build_image_prompts_prompt(plan: Dict[str, Any], parts: Dict[int, PartResult]) -> str:
    scripts = [f"### PARTE {number}\n{parts[number].roteiro}" for number in sorted(parts)]
    return f"""
PROMPTS DE IMAGEM
Responda normal, sem JSON e sem markdown.

Crie prompts de imagem somente para momentos que aparecem DENTRO das conversas aprovadas.
Nao gere cenas separadas. Nao gere prompts de video. Nao invente momentos fora do chat.

Plano aprovado:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Conversas aprovadas:
{truncate_text(chr(10).join(scripts), 24000)}

Regras:
- Priorize linhas FOTO:, prints, provas, objetos, mensagens no celular e momentos de alta emocao.
- Cada prompt deve apontar a parte e o trecho exato da conversa que inspirou a imagem.
- Prompts em ingles, com realismo cinematografico.
- Se houver texto visivel na imagem, deixe esse texto em portugues no campo visible_text.
- Gere poucos prompts bons por parte.

Formato da resposta:
PROMPT DE IMAGEM:
Parte:
Trecho:
Imagem:
Prompt:

Repita esse bloco para cada imagem.
""".strip()
