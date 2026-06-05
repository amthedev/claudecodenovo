# Upgrade de modelo — quando o Qwen-30B não basta

Documento de decisão pro Allan. Contexto: clientes relatam que o modelo "não
obedece / interpreta mal / faz além do pedido". Já otimizamos o proxy ao máximo
(temperatura 0.3, prompt de obediência reorganizado, contenção de escopo, modo
conteúdo sem injeções). O que sobra é **teto do modelo**: Qwen3-Coder-30B é bom
mas obediência fina e interpretação de texto são onde um 30B fica atrás de
Opus/Sonnet. Este doc é pra quando você decidir investir em modelo melhor.

> ⚠️ Preços de GPU mudam toda hora. Os valores aqui são ORDEM DE GRANDEZA
> (meados de 2026). Confirme o preço atual no RunPod/Vast na hora de decidir.

## Stack atual (baseline)

- Modelo: **Qwen2.5-Coder-32B-Instruct-AWQ**, servido como `qwen2.5-coder-32b`
- GPU: **RTX 6000 Ada 48GB** (RunPod), com `--cpu-offload-gb 30 --swap-space 30`
- Contexto: 30000 tokens (proxy). Qwen2.5-Coder-32B só tem **32768 nativo** (config.json), pod roda `--max-model-len 32768`. NÃO usar 65536 (o vLLM recusa; forçar com VLLM_ALLOW_LONG_MAX_MODEL_LEN corrompe a saída)
- Reserva de output: 8192 (era 12288). Sobra ~22k pra conversa/histórico. Ajustável via VLLM_CONTEXT_OUTPUT_RESERVE
- Flags tool calling (Qwen2.5 usa parser **hermes**, NÃO qwen3_coder):
  `--enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching`
- Qwen2.5 NÃO tem modo thinking (era do Qwen3). O proxy só injeta `enable_thinking`
  quando o nome do modelo contém "qwen3" (ou VLLM_THINKING_SUPPORTED=on).
- Custo aproximado: ~$0.80–1.20/h (RTX 6000 Ada)

## Opções de upgrade (da mais barata pra melhor)

### Opção A — Qwen2.5-72B-Instruct (AWQ) — melhor custo/obediência
- **Por quê**: 72B obedece e interpreta MUITO melhor que 30B. É o maior salto de
  qualidade por dólar. Versão Instruct (não-coder) é mais equilibrada pra
  obediência geral; existe Qwen2.5-Coder-32B se o foco for só código.
- **VRAM**: ~48GB em AWQ 4-bit (cabe numa A100 80GB ou 2× RTX 6000 Ada com folga
  pra contexto maior). Numa única 48GB fica apertado com pouco contexto.
- **GPU sugerida**: **A100 80GB** (~$1.50–2.50/h) ou **H100 80GB** (~$2.50–4/h).
- **Contexto possível**: 64k–128k com 80GB (vs 43k hoje).
- **Ganho esperado**: obediência e interpretação claramente melhores; some a
  maioria dos "fez além do pedido / não entendeu".

### Opção B — Qwen2.5-72B sem quantização / modelo maior — qualidade alta
- Precisa de 2× A100 80GB ou 1× H200 141GB. Custo ~$4–8/h.
- Ganho marginal sobre o AWQ 72B não compensa o custo no seu caso. **Pular**,
  a não ser que precise de contexto gigante (200k+).

### Opção C — Rotear casos críticos pro Claude real — qualidade máxima sob demanda
- **Não troca o vLLM.** Em vez disso, clientes que precisam de obediência fina
  usam Claude de verdade (Anthropic ou via OpenRouter); coding simples continua
  no Qwen barato.
- **Como**: o proxy já suporta provider `anthropic` e `openrouter`. Daria pra criar
  uma chave/alias que aponta pro Claude real pra esses clientes. Custo: por token
  (caro por request, mas zero infra fixa).
- **Quando faz sentido**: poucos clientes exigentes, muitos clientes casuais.
  Você paga Claude só pra quem precisa.

## Recomendação

1. **Se quer um único upgrade que resolve a maioria**: Opção A (Qwen2.5-72B AWQ
   numa A100 80GB). Melhor relação custo/obediência. ~2× o custo de GPU atual,
   mas resolve o problema de raiz.
2. **Se os clientes exigentes são poucos**: Opção C (rotear pro Claude real só
   eles), mantendo o Qwen pro resto. Zero mudança de infra, paga por uso.
3. **Não vale**: Opção B (72B full / H200) — custo alto, ganho marginal.

## Como migrar (Opção A) — passos

1. Subir pod RunPod com A100 80GB.
2. Comando vLLM (começar DIRETO em `--model`, a imagem já tem `vllm serve`):
   ```
   --model Qwen/Qwen2.5-72B-Instruct-AWQ --served-model-name qwen25-72b
   --gpu-memory-utilization 0.95 --max-model-len 65536 --max-num-seqs 8
   --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching
   --trust-remote-code
   ```
   (72B não-coder usa `--tool-call-parser hermes`; sem `--reasoning-parser` se não
   for um modelo de raciocínio.)
3. No proxy/Square Cloud: atualizar a URL via painel admin (`VLLM_ENDPOINT_FILE`)
   ou `HOSTED_VLLM_API_BASE`, e `HOSTED_VLLM_MODELS=["qwen25-72b"]` +
   `ANTHROPIC_DEFAULT_*_MODEL=hosted_vllm/qwen25-72b`.
4. Subir `VLLM_MODEL_CONTEXT` no proxy pro novo teto (ex: 65536) pra a compactação
   parar de disparar tão cedo.
5. Testar obediência com os casos reais dos clientes antes de anunciar.

## Lembrete sobre o proxy

Tudo que dava pra fazer no software já foi feito (commits da sessão de jun/2026):
temperatura 0.3, prompt de obediência em 5 regras numeradas, contenção de escopo,
confirmação de destrutivo, obedecer reverter, modo conteúdo sem injeções. Se
ainda houver reclamação de obediência DEPOIS de tudo isso, é modelo — este doc é
o caminho.
