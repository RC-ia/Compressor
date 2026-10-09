# Compressor — reconstrução aproximada por representantes compartilhados

Experimento de pesquisa para testar se milhares ou milhões de pesos podem ser substituídos por um conjunto menor de valores representativos. Os pesos originais individuais deixam de ser armazenados: a reconstrução usa um **codebook de representantes** e um **índice por posição**.

O alvo inicial é `Qwen/Qwen3.5-4B` em Safetensors. O código não converte para GGUF e não instancia o modelo Transformers completo para fazer a análise.

## Ideia

Para cada peso original (w_i), o agrupamento seleciona um representante (c_j). A reconstrução aproxima o peso usando (hat w_i=c_j). Com (K) representantes, um índice precisa identificar qual dos (K) valores pertence a cada posição. Em princípio são necessários (lceillog_2 Kceil) bits por índice, além do próprio codebook e dos metadados.

Isso não é uma compressão sem custo de informação: o mapa de associações é essencial para saber onde cada representante deve ser usado. A hipótese a testar é que esse mapa seja muito menor que armazenar cada peso em 16 bits, com uma perda aceitável.

## O que o analisador faz

- Lê arquivos Safetensors em blocos, sem carregar o modelo inteiro na RAM.
- Conta padrões binários distintos em BF16/FP16 e, quando possível, F32.
- Faz amostragem estratificada por tensor: mantém cobertura de tensores pequenos e distribui o restante da amostra por tamanho.
- Divide a amostra de cada tensor em conjuntos separados de treino e validação.
- Testa três estratégias de codebook:
  - **global**: um codebook para todos os tensores;
  - **layer**: codebook compartilhado por camada do backbone; embeddings, normas finais e tensores sem índice de camada ficam isolados;
  - **tensor**: codebook independente por tensor.
- Ajusta representantes com k-means escalar ponderado, usando inicializações alternativas para reduzir falhas em distribuições assimétricas e evitando descartar boas sementes de quantis quando ocorrem duplicatas.
- Os pesos da amostra compensam a cobertura mínima por tensor, para que tensores pequenos não dominem artificialmente o ajuste.
- Avalia o erro em dados mantidos fora do ajuste, ponderado pelo número real de parâmetros de cada tensor.
- Estima armazenamento com índices empacotados e codebooks FP32, FP16 ou BF16.

## Instalação

Python 3.10+ recomendado.

```bash
pip install -r requirements.txt
```

## Execução no Qwen

```bash
python analyze_weights.py --model Qwen/Qwen3.5-4B --sample-size 500000
```

O primeiro uso baixa do Hugging Face os arquivos `.safetensors` e metadados necessários, consumindo vários GB de espaço. A análise é executada na CPU e pode levar um tempo considerável para testar todos os grupos.

## Opções

```bash
# Aumentar amostra e cobertura mínima por tensor
python analyze_weights.py --model Qwen/Qwen3.5-4B \
  --sample-size 1000000 --min-samples-per-tensor 256

# Comparar somente codebook global e por camada (execução mais curta)
python analyze_weights.py --model Qwen/Qwen3.5-4B \
  --scopes global,layer --groups 4,8,16,32,64,128,256,512,1024

# Testar codebooks BF16 para reduzir o custo de armazenamento dos representantes
python analyze_weights.py --model Qwen/Qwen3.5-4B \
  --codebook-dtype bf16 --groups 4,16,64,256,1024

# Usar arquivos de checkpoint já baixados
python analyze_weights.py --model ./Qwen3.5-4B --output-dir resultados

# Controlar a quantidade aproximada de elementos lidos por bloco
python analyze_weights.py --model Qwen/Qwen3.5-4B --chunk-elements 250000
```

Argumentos principais:

| Argumento | Padrão | Significado |
|---|---|---|
| `--sample-size` | 150000 | Número de pesos amostrados para análise |
| `--min-samples-per-tensor` | 128 | Cobertura mínima de cada tensor, sujeita ao orçamento total |
| `--validation-fraction` | 0.2 | Fração da amostra separada para validação |
| `--groups` | 2 a 4096 | Número máximo de representantes por codebook |
| `--scopes` | `global,layer,tensor` | Estratégias de compartilhamento a comparar |
| `--codebook-dtype` | `fp32` | Precisão usada para armazenar os representantes |

## Resultados

O diretório `compressor_results/` conterá:

- `summary.json`: metadados, estatísticas globais, contagem de padrões exatos e resultados agregados.
- `group_analysis.csv`: uma linha por estratégia e quantidade de grupos, com RMSE/MAE de validação, bits por índice e custo estimado.
- `tensor_stats.csv`: estatísticas de valores por tensor/camada.

Campos que merecem atenção em `group_analysis.csv`:

- **`scope`**: estratégia global, por camada ou por tensor.
- **`requested_groups_per_codebook`**: máximo solicitado para cada tabela de representantes.
- **`actual_groups_total_across_codebooks`**: soma dos representantes efetivamente utilizados em todas as tabelas.
- **`validation_rmse_population_weighted`**: erro quadrático médio na validação, ponderado pelo número de parâmetros do tensor.
- **`validation_rmse_over_full_model_weight_std`**: RMSE de validação dividido pelo desvio-padrão dos pesos do modelo completo.
- **`weighted_mean_index_bits_per_parameter`**: custo médio estimado dos índices.
- **`estimated_total_MB_ideal_packed`**: índices mais codebooks, em um empacotamento ideal, antes de cabeçalhos e outros metadados.
- **`groups_capped_by_sample_uniques`**: codebooks que não puderam chegar ao número solicitado porque a amostra de treino daquele grupo tinha menos valores distintos.

## Como interpretar sem tirar conclusões precipitadas

1. Pouco erro numérico nos pesos não garante a mesma perplexidade, logits ou qualidade de resposta.
2. O codebook global tende a ser mais barato, mas pode misturar distribuições muito diferentes. Codebooks locais podem diminuir o erro, ao custo de tabelas extras.
3. O codebook por tensor pode ficar limitado pela quantidade de amostras disponíveis de tensores pequenos. Se muitos codebooks forem limitados, aumente `--sample-size` e `--min-samples-per-tensor`.
4. As taxas de armazenamento são estimativas, não tamanho medido de um arquivo final. Incluem índices idealmente empacotados por tensor e representantes no dtype escolhido, mas não cabeçalhos, alinhamento, descritores nem overhead de carregamento.
5. A reconstrução envolve lookup de representantes. O custo durante inferência e a compatibilidade com kernels de GPU/CPU precisam ser medidos separadamente; não está garantido que a execução fique mais rápida.
6. O analisador não modifica nem regrava o checkpoint original.

## Próxima etapa experimental

Escolher uma ou duas configurações com boa relação erro/tamanho, implementar encoder e decoder de verdade, reconstruir uma ou duas matrizes, verificar o erro diretamente em todos os seus pesos e comparar as saídas de camadas. Só depois avaliar o modelo completo com perplexidade/logits e inferência real.
