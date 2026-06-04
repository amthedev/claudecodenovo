// Teste do parser de roteiro → mensagens da tela de WhatsApp.
// Roda em Node sem browser: extrai só a função pura do app.js e avalia.
// Uso: node tests/test_bote_parser.js  (sai 0 se ok, 1 se falha)
const fs = require("fs");
const path = require("path");

const code = fs.readFileSync(
  path.join(__dirname, "..", "src", "proxy_app", "bote", "app.js"),
  "utf8"
);
const match = code.match(/function parseRoteiroToMessages[\s\S]*?\n}/);
if (!match) {
  console.error("FALHA: não achei parseRoteiroToMessages no app.js");
  process.exit(1);
}
// eslint-disable-next-line no-eval
eval(match[0]);

let failures = 0;
function assert(cond, msg) {
  if (!cond) { console.error("✗ " + msg); failures++; }
  else { console.log("✓ " + msg); }
}

// 1) básico: contato vs eu, divisor, foto
const r1 = "[Hoje]\nDébora: oi sumida\nJana: que foto??\nFOTO: PRINT\nJana: nossa";
const m1 = parseRoteiroToMessages(r1, "Débora");
assert(m1.length === 5, "5 itens parseados");
assert(m1[0].kind === "date" && m1[0].text === "Hoje", "divisor de data");
assert(m1[1].kind === "theirs", "Débora (contato) = theirs");
assert(m1[2].kind === "mine", "Jana = mine");
assert(m1[3].photo === true, "FOTO vira bolha de foto");

// 2) sem nome de contato: primeiro falante vira o contato
const r2 = "Ana: primeiro\nBeto: segundo\nAna: terceiro";
const m2 = parseRoteiroToMessages(r2, "");
assert(m2[0].kind === "theirs", "primeiro falante = theirs quando sem contato");
assert(m2[1].kind === "mine", "segundo falante = mine");
assert(m2[2].kind === "theirs", "primeiro falante consistente");

// 3) linhas vazias e lixo são ignoradas
const r3 = "\n\nAna: oi\n\n";
const m3 = parseRoteiroToMessages(r3, "Ana");
assert(m3.length === 1, "linhas vazias ignoradas");

// 4) roteiro vazio → lista vazia
assert(parseRoteiroToMessages("", "X").length === 0, "vazio → []");

// 5) "Eu sou" define o lado: o nome em meName = mine, resto = theirs
const r5 = "Mariana: oi\nCarlos: oi tudo bem\nLarissa: e ai\nMariana: to bem";
const m5 = parseRoteiroToMessages(r5, "Carlos", "Mariana");
assert(m5[0].kind === "mine", "Mariana (eu sou) = mine");
assert(m5[1].kind === "theirs", "Carlos (contato) = theirs");
assert(m5[2].kind === "theirs", "Larissa (terceiro) = theirs");
assert(m5[3].kind === "mine", "Mariana de novo = mine");
// terceiro (Larissa) recebe prefixo do nome pra distinguir
assert(m5[2].text.startsWith("Larissa:"), "terceiro prefixado com nome");
// contato principal (Carlos) NÃO recebe prefixo
assert(!m5[1].text.startsWith("Carlos:"), "contato principal sem prefixo");

// 6) remove repetições do loop. Falas LONGAS que repetem = loop → corta na 1ª
//    repetição. Falas curtas toleram 2 (pra "sim"/"kkkk" legítimos não sumirem).
const r6 = "Ana: É só uma foto? kkkkk\nBeto: É, é só uma foto\n" +
           "Ana: É só uma foto? kkkkk\nBeto: É, é só uma foto\n" +
           "Ana: É só uma foto? kkkkk\nBeto: É, é só uma foto";
const m6 = parseRoteiroToMessages(r6, "Beto", "Ana");
assert(m6.length === 2, "loop de falas longas: 3+3 → 2 (1 de cada)");

// 6b) falas curtas comuns toleram repetir (até 2)
const r6b = "Ana: sim\nAna: sim\nAna: sim";
const m6b = parseRoteiroToMessages(r6b, "Ana", "");
assert(m6b.length === 2, "fala curta repetida tolera 2 (3 → 2)");

if (failures) { console.error(`\n${failures} teste(s) falharam`); process.exit(1); }
console.log("\nTodos os testes do parser passaram.");
