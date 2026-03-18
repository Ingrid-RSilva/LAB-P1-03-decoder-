# Transformer Decoder — Simulação Pedagógica em NumPy

> **Aviso:** Partes complementadas com IA, revisadas por Ingrid.

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura Implementada](#arquitetura-implementada)
- [Estrutura do Código](#estrutura-do-código)
- [Pré-requisitos](#pré-requisitos)
- [Como Executar](#como-executar)
- [Tarefas Implementadas](#tarefas-implementadas)
- [Hiperparâmetros](#hiperparâmetros)
- [Limitações e Avisos](#limitações-e-avisos)
- [Referências](#referências)

---

## Visão Geral

Este projeto simula o pipeline de inferência de um Transformer encoder-decoder (arquitetura do paper *"Attention is All You Need"*, Vaswani et al., 2017), com foco no comportamento do Decoder durante a geração auto-regressiva de texto.

**Caso de uso ilustrativo:** tradução de francês → inglês (os dados são fictícios/aleatórios; o foco é a mecânica, não a qualidade da tradução).

---

## Arquitetura Implementada

```
Entrada (tokens do Encoder)
        │
        ▼
┌───────────────────┐
│  Encoder (6 cam.) │  ← Self-Attention + FFN + Add&Norm
└────────┬──────────┘
         │  encoder_output  [batch, seq_enc, d_model]
         │
         ▼
┌──────────────────────────────────────────┐
│              Decoder (1 passo)           │
│                                          │
│  Token(s) gerados até agora              │
│       │                                  │
│       ▼                                  │
│  Embedding + Positional Encoding         │
│       │                                  │
│       ▼                                  │
│  Masked Self-Attention (máscara causal)  │
│       │                                  │
│       ▼                                  │
│  Cross-Attention ◄── encoder_output      │
│       │                                  │
│       ▼                                  │
│  Projeção Linear → Softmax → probs       │
└──────────────────────────────────────────┘
         │
         ▼
   Próximo token (top-k sampling)
```

---

## Estrutura do Código

```
transformer_decoder.py
│
├── Hiperparâmetros globais
│   └── d_model, vocab_size, vocabulário fictício
│
├── Funções auxiliares
│   ├── softmax()          — versão numericamente estável
│   └── layer_norm()       — Layer Normalization
│
├── TAREFA 1 — Máscara Causal
│   ├── create_causal_mask(seq_len)
│   └── Prova: pesos de atenção futuros são estritamente 0.0
│
├── TAREFA 2 — Cross-Attention (ponte Encoder-Decoder)
│   ├── scaled_dot_product_attention(Q, K, V, mask)
│   └── cross_attention(encoder_out, decoder_state)
│
└── TAREFA 3 — Loop de Inferência Auto-Regressivo
    ├── generate_next_token(current_sequence, encoder_out)
    └── Loop while: gera tokens até <EOS> ou max_steps
```

---

## Pré-requisitos

- Python 3.8+
- NumPy

```bash
pip install numpy
```

---

## Como Executar

```bash
python transformer_decoder.py
```

A saída no terminal percorre as três tarefas em sequência, exibindo shapes de tensores, verificações e a frase final gerada token a token.

**Exemplo de saída (Tarefa 3):**

```
Iniciando geração auto-regressiva...
  Passo  1 | token escolhido: 'palavra_7342' (id=7342, prob=0.0021)
  Passo  2 | token escolhido: 'palavra_512'  (id=512,  prob=0.0019)
  ...
  Passo  5 | token escolhido: '<EOS>'        (id=1,    prob=1.0000)

✓ Token <EOS> detectado. Geração encerrada.

FRASE FINAL GERADA:
<START> palavra_7342 palavra_512 ... <EOS>
```

> Os tokens intermediários variam a cada execução (gerador sem seed fixa).

---

## Tarefas Implementadas

### Tarefa 1 — Máscara Causal (Look-Ahead Mask)

Garante que a posição `i` **nunca veja posições futuras** `i+1, i+2, ...` durante o Self-Attention do Decoder. Implementada como uma matriz triangular superior preenchida com `-inf`, que após o Softmax resulta em pesos estritamente iguais a `0.0`.

```
Máscara 5×5:
[[  0.  -inf  -inf  -inf  -inf]
 [  0.    0.  -inf  -inf  -inf]
 [  0.    0.    0.  -inf  -inf]
 [  0.    0.    0.    0.  -inf]
 [  0.    0.    0.    0.    0.]]
```

### Tarefa 2 — Cross-Attention

A "ponte" entre Encoder e Decoder:

| Componente | Origem |
|---|---|
| Query (Q) | Estado atual do Decoder |
| Key (K) | Saída do Encoder |
| Value (V) | Saída do Encoder |

Cada token já gerado pode "consultar" **todos** os tokens da frase de entrada, sem máscara.

### Tarefa 3 — Geração Auto-Regressiva

Loop que repete até `<EOS>` ou `max_steps = 20`:

1. Embedding + Positional Encoding dos tokens gerados até agora
2. Masked Self-Attention
3. Cross-Attention com `encoder_output`
4. Projeção linear → Softmax
5. **Top-K Sampling** (`k = 50`) para selecionar o próximo token
6. Concatena o token escolhido à sequência e repete

---

## Hiperparâmetros

| Parâmetro | Valor | Descrição |
|---|---|---|
| `d_model` | 512 | Dimensão dos embeddings |
| `vocab_size` | 10 000 | Tamanho do vocabulário fictício |
| `seq_len_frances` | 10 | Tokens na frase do Encoder |
| `encoder_layers` | 6 | Camadas do Encoder simulado |
| `top_k` | 50 | Top-K para sampling |
| `temperatura` | 2.0 | Temperatura do Softmax final |
| `max_steps` | 20 | Limite de tokens gerados |

---

## Limitações e Avisos

> **Este é um mock pedagógico**, não um modelo treinado.

- Todos os pesos são **aleatórios** — a frase gerada não tem significado linguístico.
- A parada em `<EOS>` no passo 5 é **forçada manualmente** para fins de demonstração.
- O ruído adicionado aos logits (`× 50`) existe para simular variabilidade real, já que pesos aleatórios tendem a concentrar probabilidade num único token.
- Multi-head attention, FFN no Decoder e múltiplas camadas do Decoder **não estão implementados** — o foco é na mecânica das três tarefas centrais.

---
- Vaswani, A. et al. **Attention Is All You Need**. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP
