import numpy as np

# HIPERPARÂMETROS GLOBAIS
d_model = 512
vocab_size = 10_000  # Vocabulário fictício com V = 10.000

# Vocabulário fictício: índice -> palavra
id2word = {i: f"palavra_{i}" for i in range(vocab_size)}
id2word[0] = "<START>"
id2word[1] = "<EOS>"
id2word[2] = "Ele"
id2word[3] = "fortalece"
id2word[4] = "o"
id2word[5] = "cansado"

# Mapeamento inverso: palavra -> índice
word2id = {v: k for k, v in id2word.items()}



# FUNÇÕES AUXILIARES

def softmax(x):
    """
    Calcula a função softmax de forma numericamente estável.
    
    Args:
        x: Array de entrada
    
    Returns:
        Array com probabilidades (soma = 1 ao longo do último eixo)
    """
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def layer_norm(x, eps=1e-6):
    """
    Aplica Layer Normalization.
    
    Args:
        x: Tensor de entrada
        eps: Epsilon para estabilidade numérica
    
    Returns:
        Tensor normalizado
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# TAREFA 1: MÁSCARA CAUSAL (LOOK-AHEAD MASK)
print("TAREFA 1: Implementando a Máscara Causal (Look-Ahead Mask)")


def create_causal_mask(seq_len):
    """
    Cria uma máscara causal (look-ahead mask) para o Decoder.
    
    A máscara impede que a posição i atenda à posição i+1 (futuro).
    - Parte triangular inferior (incluindo diagonal): 0 (permite atenção)
    - Parte triangular superior: -infinito (bloqueia atenção)
    
    Args:
        seq_len: Tamanho da sequência
    
    Returns:
        Matriz [seq_len, seq_len] com a máscara causal
    """
    # Cria matriz de zeros
    mask = np.zeros((seq_len, seq_len))
    
    # Preenche a parte triangular superior (k=1 significa acima da diagonal) com -infinito
    mask = np.where(
        np.triu(np.ones((seq_len, seq_len)), k=1) == 1,
        -np.inf,
        0
    )
    
    return mask


# Demonstração com seq_len = 5
seq_len_prova = 5
M = create_causal_mask(seq_len_prova)

print(f"\nMáscara Causal M ({seq_len_prova}x{seq_len_prova}):")
print(M)
print("\nExplicação:")
print("  - 0.0 nas posições permitidas (diagonal e abaixo)")
print("  - -inf nas posições bloqueadas (acima da diagonal)")

# PROVA REAL: Testar que posições futuras têm probabilidade 0
print("\n--- PROVA REAL ---")

# Criar matrizes Q e K fictícias
Q_prova = np.random.randn(seq_len_prova, d_model)
K_prova = np.random.randn(seq_len_prova, d_model)

# Calcular scores de atenção: (Q @ K^T) / sqrt(d_k)
scores_sem_mask = Q_prova @ K_prova.T / np.sqrt(d_model)
print(f"\nScores de Atenção (Q @ K^T / sqrt(d_k)) ANTES da máscara:")
print(np.round(scores_sem_mask, 4))

# Adicionar a máscara causal
scores_com_mask = scores_sem_mask + M
print(f"\nScores de Atenção APÓS adicionar a máscara M:")
print(np.round(scores_com_mask, 4))

# Aplicar Softmax
pesos_atencao = softmax(scores_com_mask)
print("\nPesos de Atenção após Softmax:")
print(np.round(pesos_atencao, 4))

# Verificar que posições futuras (triângulo superior) são estritamente 0.0
mascara_superior = np.triu(np.ones((seq_len_prova, seq_len_prova), dtype=bool), k=1)
posicoes_futuras_zeradas = np.all(pesos_atencao[mascara_superior] == 0.0)

print(f"\n✓ VERIFICAÇÃO: Probabilidades das palavras futuras são estritamente 0.0? {posicoes_futuras_zeradas}")
print("  (O triângulo superior da matriz de pesos deve conter apenas zeros)")


# TAREFA 2: A PONTE ENCODER-DECODER (CROSS-ATTENTION)
print("TAREFA 2: A Ponte Encoder-Decoder (Cross-Attention)")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calcula o Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Queries [batch, seq_len_q, d_k]
        K: Keys [batch, seq_len_k, d_k]
        V: Values [batch, seq_len_k, d_v]
        mask: Máscara opcional [seq_len_q, seq_len_k]
    
    Returns:
        Saída da atenção [batch, seq_len_q, d_v]
    """
    d_k = Q.shape[-1]
    
    # Calcular scores: Q @ K^T / sqrt(d_k)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    
    # Aplicar máscara se fornecida
    if mask is not None:
        scores = scores + mask
    
    # Softmax para obter pesos de atenção
    weights = softmax(scores)
    
    # Multiplicar pelos Values
    return weights @ V


# 1. Criar tensor fictício para saída do Encoder
batch_size = 1
seq_len_frances = 10  # Frase em francês com 10 tokens

print(f"\n1. Simulando saída do Encoder:")
print(f"   Frase fictícia do Encoder (francês): {[f'token_enc_{i}' for i in range(seq_len_frances)]}")

# Simular passagem por 6 camadas do Encoder
X = np.random.randn(batch_size, seq_len_frances, d_model)
print(f"   Shape inicial: {X.shape}")

encoder_output = X.copy()
for layer_idx in range(6):
    # Pesos aleatórios para cada camada
    WQ_ = np.random.randn(d_model, d_model)
    WK_ = np.random.randn(d_model, d_model)
    WV_ = np.random.randn(d_model, d_model)
    W1_ = np.random.randn(d_model, d_model * 4)
    b1_ = np.zeros(d_model * 4)
    W2_ = np.random.randn(d_model * 4, d_model)
    b2_ = np.zeros(d_model)
    
    # Self-Attention
    Q_ = encoder_output @ WQ_
    K_ = encoder_output @ WK_
    V_ = encoder_output @ WV_
    scores_ = Q_ @ K_.transpose(0, 2, 1) / np.sqrt(d_model)
    X_att = softmax(scores_) @ V_
    
    # Add & Norm
    X_n1 = layer_norm(encoder_output + X_att)
    
    # Feed-Forward Network
    X_ffn = np.maximum(0, X_n1 @ W1_ + b1_) @ W2_ + b2_
    
    # Add & Norm
    encoder_output = layer_norm(X_n1 + X_ffn)

print(f"   Shape após 6 camadas do Encoder: {encoder_output.shape}")

# 2. Criar tensor fictício para estado do Decoder
seq_len_ingles = 4  # Decoder já gerou 4 tokens em inglês

decoder_state = np.random.randn(batch_size, seq_len_ingles, d_model)
print(f"\n2. Simulando estado do Decoder:")
print(f"   Tokens já gerados (inglês): {[f'token_dec_{i}' for i in range(seq_len_ingles)]}")
print(f"   decoder_state shape: {decoder_state.shape}")

# 3. Implementar Cross-Attention
print("\n3. Implementando Cross-Attention:")

# Matrizes de projeção para Cross-Attention
WQ_cross = np.random.randn(d_model, d_model)
WK_cross = np.random.randn(d_model, d_model)
WV_cross = np.random.randn(d_model, d_model)


def cross_attention(encoder_out, decoder_state):
    """
    Realiza Cross-Attention entre Encoder e Decoder.
    
    - Query (Q) vem do Decoder (estado atual da geração)
    - Keys (K) e Values (V) vêm do Encoder (memória da frase original)
    
    Args:
        encoder_out: Saída do Encoder [batch, seq_len_encoder, d_model]
        decoder_state: Estado do Decoder [batch, seq_len_decoder, d_model]
    
    Returns:
        Saída da Cross-Attention [batch, seq_len_decoder, d_model]
    """
    # Query vem do Decoder
    Q = decoder_state @ WQ_cross
    
    # Keys e Values vêm do Encoder
    K = encoder_out @ WK_cross
    V = encoder_out @ WV_cross
    
    # Scaled Dot-Product Attention SEM máscara
    # (Decoder pode ver toda a frase do Encoder)
    return scaled_dot_product_attention(Q, K, V, mask=None)


# 4. Calcular Cross-Attention
cross_out = cross_attention(encoder_output, decoder_state)

print(f"   Saída da Cross-Attention shape: {cross_out.shape}")
print(f"   (batch={batch_size}, seq_ingles={seq_len_ingles}, d_model={d_model})")
print("\n   ✓ Cada token do Decoder agora carrega informação de TODA a frase do Encoder!")
print(f"\n   Primeiros 6 valores do token [0] do Decoder após Cross-Attention:")
print(f"   {np.round(cross_out[0, 0, :6], 4)}")

# TAREFA 3: LOOP DE INFERÊNCIA AUTO-REGRESSIVO
print("TAREFA 3: Simulando o Loop de Inferência Auto-Regressivo")

# Usar gerador aleatório SEM seed fixa para resultados diferentes a cada execução
rng3 = np.random.default_rng()

# Tabela de embeddings para o Decoder
embedding_table_dec = rng3.standard_normal((vocab_size, d_model))

# Pesos fixos para o Decoder (simulação)
WQ_self_fixed = rng3.standard_normal((d_model, d_model))
WK_self_fixed = rng3.standard_normal((d_model, d_model))
WV_self_fixed = rng3.standard_normal((d_model, d_model))
WQ_cross_t3 = rng3.standard_normal((d_model, d_model))
WK_cross_t3 = rng3.standard_normal((d_model, d_model))
WV_cross_t3 = rng3.standard_normal((d_model, d_model))
W_proj = rng3.standard_normal((d_model, vocab_size))  # Projeção para vocabulário


def generate_next_token(current_sequence, encoder_out):
    """
    Simula um passo completo do Decoder para gerar o próximo token.
    
    Fluxo:
        1. Embedding dos tokens já gerados
        2. Positional Encoding (sin/cos do paper "Attention is All You Need")
        3. Masked Self-Attention (com máscara causal)
        4. Cross-Attention com a saída do Encoder
        5. Projeção linear para o tamanho do vocabulário
        6. Softmax para obter distribuição de probabilidades

    Returns:
        Vetor de probabilidades de tamanho vocab_size (10.000,)
    """
    seq_len_dec = len(current_sequence)
    
    # 1. EMBEDDING: Buscar na tabela de embeddings pelo id de cada token
    ids = [word2id.get(tok, 0) for tok in current_sequence]
    dec_emb = embedding_table_dec[ids][np.newaxis, :, :]  # (1, seq_len_dec, d_model)
    
    # 2. POSITIONAL ENCODING: sin/cos do paper "Attention is All You Need"
    pe = np.zeros((seq_len_dec, d_model))
    for pos in range(seq_len_dec):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    dec_emb = dec_emb + pe[np.newaxis, :, :]
    
    # 3. MASKED SELF-ATTENTION (com máscara causal)
    Q_self = dec_emb @ WQ_self_fixed
    K_self = dec_emb @ WK_self_fixed
    V_self = dec_emb @ WV_self_fixed
    mask = create_causal_mask(seq_len_dec)
    self_att_out = scaled_dot_product_attention(Q_self, K_self, V_self, mask=mask)
    
    # 4. CROSS-ATTENTION com memória do Encoder
    Q_c = self_att_out @ WQ_cross_t3
    K_c = encoder_out @ WK_cross_t3
    V_c = encoder_out @ WV_cross_t3
    cross_att_out = scaled_dot_product_attention(Q_c, K_c, V_c, mask=None)
    
    # 5. Pegar o vetor da ÚLTIMA posição (token mais recente)
    ultimo_vetor = cross_att_out[0, -1, :]  # (d_model,)
    
    # 6. PROJEÇÃO LINEAR para vocabulário + Softmax com RUÍDO SIGNIFICATIVO
    logits = ultimo_vetor @ W_proj  # (vocab_size,)
    
    # Adicionar ruído aleatório GRANDE aos logits para simular variabilidade real
    # Como este é um MOCK pedagógico (pesos aleatórios), precisamos de mais ruído
    # para que a distribuição não fique concentrada em um único token
    ruido = np.random.randn(vocab_size) * 50  # Ruído grande para variabilidade
    logits_com_ruido = logits * 0.1 + ruido   # Reduzir peso dos logits originais
    
    # Aplicar temperatura para controlar a suavidade da distribuição
    temperatura = 2.0
    logits_final = logits_com_ruido / temperatura
    
    probs = softmax(logits_final.reshape(1, -1)).flatten()
    
    return probs


# LOOP AUTO-REGRESSIVO
print("\nIniciando geração auto-regressiva...")


# Começar com o token especial <START>
sequencia_gerada = ["<START>"]
max_steps = 20  # Limite máximo de tokens para evitar loop infinito

step = 0
while step < max_steps:
    step += 1
    
    # Gerar distribuição de probabilidades para o próximo token
    probs = generate_next_token(sequencia_gerada, encoder_output)
    
    # MOCK PEDAGÓGICO: Forçar <EOS> no passo 5 para demonstrar parada
    # Em um modelo real, isso seria determinado pelas probabilidades aprendidas
    if step == 5:
        probs = np.zeros(vocab_size)
        probs[1] = 1.0  # id=1 → <EOS>
    
    # TOP-K SAMPLING: Selecionar aleatoriamente entre os k tokens mais prováveis
    # Isso garante variabilidade mantendo coerência com a distribuição
    top_k = 50
    top_k_indices = np.argsort(probs)[-top_k:]  # Índices dos top-k mais prováveis
    top_k_probs = probs[top_k_indices]
    top_k_probs = top_k_probs / top_k_probs.sum()  # Renormalizar
    
    # Amostrar dos top-k
    escolhido_idx = np.random.choice(len(top_k_indices), p=top_k_probs)
    proximo_id = int(top_k_indices[escolhido_idx])
    proximo_token = id2word[proximo_id]
    
    print(f"  Passo {step:2d} | token escolhido: '{proximo_token}' "
          f"(id={proximo_id}, prob={probs[proximo_id]:.4f})")
    
    # Adicionar o novo token à sequência
    sequencia_gerada.append(proximo_token)
    
    # CONDIÇÃO DE PARADA: Se o token for <EOS>, encerrar
    if proximo_token == "<EOS>":
        print("\n✓ Token <EOS> detectado. Geração encerrada.")
        break

# Exibir frase final
print("\nFRASE FINAL GERADA:")
print(" ".join(sequencia_gerada))
