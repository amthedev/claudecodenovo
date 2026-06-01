# Instalador da API no Antigravity

Este utilitario configura a extensao oficial Claude Code dentro do Antigravity para
usar o endpoint Anthropic compativel deste proxy. Ele funciona em Windows, macOS e
Linux sem alterar os arquivos internos do IDE.

## Uso

Com Python 3 instalado:

```bash
python installer.py
```

Na tela, informe a URL base e a chave da API entregues ao usuario. A URL deste
projeto ja aparece preenchida.

O instalador:

1. preserva e atualiza `~/.claude/settings.json`;
2. cria um backup antes de sobrescrever um JSON existente;
3. atualiza as preferencias do Antigravity quando encontra a instalacao;
4. tenta instalar `anthropic.claude-code` usando o comando do Antigravity;
5. nunca imprime o token na tela ou nos logs.

Depois da configuracao, feche e abra o Antigravity. Caso o IDE ainda nao estivesse
instalado, instale-o e execute esta ferramenta novamente.

## Linha de comando

O modo de linha de comando facilita suporte e testes automatizados:

```bash
python installer.py \
  --token "TOKEN_DO_USUARIO" \
  --base-url "https://claude-code-api.squareweb.app" \
  --model "claude-code-sonnet"
```

Use `--no-install-extension` para alterar somente os arquivos JSON e `--dry-run`
para validar os campos sem escrever no disco.

## Alias Claude com backend Qwen

O instalador apresenta o modelo `claude-code-sonnet` ao Claude Code. O proxy
resolve esse alias para o modelo real configurado no servidor. Para manter Qwen
como backend sem expor esse nome no instalador, configure o deploy do proxy:

```env
PROXY_DEFAULT_MODEL=hosted_vllm/qwen25-coder-32b
```

O alias altera somente o nome enviado pelo cliente. O modelo executado continua
sendo definido pelo servidor.

## Gerar executavel

Instale o PyInstaller e gere um binario no sistema desejado:

```bash
python -m pip install pyinstaller
python -m PyInstaller --onefile --windowed --name antigravity-api-installer installer.py
```

No macOS, prefira gerar o aplicativo `.app`:

```bash
python -m PyInstaller --onedir --windowed --name antigravity-api-installer installer.py
```

O executavel ou aplicativo sera criado em `dist/`. Gere uma versao em cada sistema operacional:
PyInstaller nao produz binarios de Windows, macOS e Linux a partir de uma unica
maquina.

O workflow `.github/workflows/build-antigravity-installer.yml` faz isso
automaticamente no GitHub Actions e publica tres artefatos para download.

## Limite Importante

Esta ferramenta adiciona o Claude Code ao Antigravity como extensao compativel com
VS Code. Ela nao desbloqueia o seletor interno de modelos do agente nativo do
Antigravity. Esse seletor ainda depende de funcionalidades internas instaveis do
IDE, e patches no JavaScript distribuido pelo aplicativo podem quebrar em updates.
