# Compressor — análise matemática bruta de pesos

Primeira fase do experimento: examinar diretamente os pesos Safetensors e medir se um conjunto de valores compartilhados consegue representar os parâmetros com menos armazenamento.

O projeto começa com `Qwen/Qwen3.5-4B`, mas também aceita qualquer diretório local que contenha arquivos `.safetensors`.

## O que o script mede

- Quantidade de parâmetros flutuantes e tamanho bruto estimado dos tensores.
- Contagem **exata** de padrões binários distintos em pesos BF16 e FP16, separadamente por formato.
- Estatísticas por tensor/camada: forma, mínimo, máximo, média, desvio-padrão, zeros e amostra de valores distintos.
- Codebooks globais com 2, 4, 8, 16, 32, ... grupos, comparando RMSE/MAE na amostra.
- Estimativa ideal de armazenamento usando índices compactados em bits e uma tabela de representantes FP32.

Os tensores são lidos em blocos pela API Safetensors; o modelo não é instanciado em Transformers e nenhuma GPU é necessária. O download inicial do modelo pode, contudo, exigir vários GB de espaço e internet.

## Instalação

Python 3.10+ recomendado.

```bash
pip install -r requirements.txt
```

## Execução completa no modelo do Hugging Face

```bash
python analyze_weights.py --model Qwen/Qwen3.5-4B
```

O script baixa os arquivos `.safetensors` e JSON necessários para o cache do Hugging Face, analisa os tensores e cria `compressor_results/` com:

- `summary.json`
- `group_analysis.csv`
- `tensor_stats.csv`

## Opções úteis

```bash
# Avaliar mais pesos na amostra
python analyze_weights.py --model Qwen/Qwen3.5-4B --sample-size 500000

# Testar grupos específicos
python analyze_weights.py --model Qwen/Qwen3.5-4B --groups 16,32,64,128,256,512,1024,2048,4096

# Reutilizar arquivos que já estão baixados
python analyze_weights.py --model ./Qwen3.5-4B --output-dir resultados

# Ajustar a RAM usada por bloco
python analyze_weights.py --model Qwen/Qwen3.5-4B --chunk-elements 250000
```

## Como interpretar

- **Padrões exatos distintos**: número de codificações de pesos que realmente aparecem no formato, não o número de grupos aproximados necessário para manter a qualidade.
- **Grupos**: quantidade de representantes numéricos compartilhados em um codebook escalar global.
- **RMSE relativo**: RMSE de reconstrução da amostra dividido pelo desvio-padrão dessa amostra. Menor significa menor erro numérico médio, não necessariamente menor perda de qualidade do modelo.
- **Economia estimada**: cenário ideal com índices empacotados usando `ceil(log2(K))` bits por peso, mais representantes FP32. Não inclui metadados, alinhamento, acesso aleatório nem implementação de kernels. Não é a taxa de compressão medida de um arquivo final.

## Limitações desta primeira fase

1. O codebook global junta tensores com escalas e funções diferentes. Codebooks separados por tensor, camada ou tipo de parâmetro podem melhorar a reconstrução, com custo adicional.
2. A contagem BF16/FP16 é exata no nível dos bits; a qualidade dos grupos é estimada a partir de uma amostra uniforme proporcional ao tamanho dos tensores.
3. O erro dos pesos não é suficiente para validar um modelo de linguagem ou visão-linguagem. A próxima fase precisa codificar e reconstruir os pesos e comparar saídas/perplexidade com um conjunto de avaliação.
4. Esta ferramenta não altera o checkpoint original e ainda não salva um modelo comprimido.

## Próxima etapa sugerida

Usar os CSVs para selecionar alguns valores de `K`; implementar encoder/decoder de codebook e comparar (a) codebook único global, (b) codebook por camada/tensor e (c) resíduos para os pesos que mais afetam a saída. Só então comparar tamanho real, velocidade e qualidade do modelo reconstruído.
