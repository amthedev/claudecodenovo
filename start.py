#!/usr/bin/env python3
"""
start.py
Wrapper de inicializacao do proxy.
1. Cria a chave universal sk-xxxx se ainda nao existir.
2. Inicia o servidor proxy substituindo o processo atual (os.execv).
"""

import os
import sys
from pathlib import Path

# Garante que a pasta src esteja no path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Passo 1: inicializar a chave universal
from proxy_app.init_default_key import ensure_universal_key
ensure_universal_key()

# Passo 2: substituir o processo pelo proxy (equivalente a rodar diretamente)
port = os.environ.get("PORT", "80")
proxy_script = str(Path(__file__).parent / "src" / "proxy_app" / "main.py")

os.execv(sys.executable, [
    sys.executable,
    proxy_script,
    "--host", "0.0.0.0",
    "--port", port,
])
