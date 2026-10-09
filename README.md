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

## Teste real de codificação e reconstrução de uma matriz

O script `compress_tensor.py` passa da estimativa para um teste concreto: escolhe **um tensor real** do checkpoint, aprende o codebook a partir de uma amostra, atribui um índice a cada peso da matriz, empacota os índices, grava um artefato NPZ e lê esse artefato de volta para reconstruir e medir o tensor inteiro.

Por padrão ele escolhe uma matriz bidimensional intermediária de até 20 milhões de pesos para evitar selecionar automaticamente uma matriz de embedding gigantesca. É possível escolher o tensor exato ou listar candidatos.

```bash
# Listar os maiores tensores no checkpoint local
python compress_tensor.py --model ./Qwen3.5-4B --list-tensors

# Codificar, decodificar e medir uma matriz com 256 representantes por tensor
python compress_tensor.py --model ./Qwen3.5-4B \
  --groups 256 --sample-size 250000 --codebook-dtype fp32

# Escolher o nome exato informado por --list-tensors
python compress_tensor.py --model ./Qwen3.5-4B \
  --tensor-name 'model.layers.0.self_attn.q_proj.weight' \
  --groups 256 --sample-size 250000 --codebook-dtype fp32

# Comparar codebooks de baixa precisão
python compress_tensor.py --model ./Qwen3.5-4B --groups 256 --codebook-dtype bf16
python compress_tensor.py --model ./Qwen3.5-4B --groups 256 --codebook-dtype fp16

# Também materializar o tensor reconstruído em Safetensors para inspeção
python compress_tensor.py --model ./Qwen3.5-4B \
  --groups 256 --save-reconstructed-safetensors
```

Resultados em `compressor_tensor_test/`:

- `*.npz`: codebook e mapa de índices comprimidos de fato, sem guardar os pesos originais.
- `*_report.json`: tamanho real do tensor-fonte, tamanho real do artefato no disco, economia observada, entropia/frequências dos índices, erro calculado sobre **todos** os pesos do tensor e teste (W X).

O codificador compara três variantes reais e escolhe a que gerar o menor artefato: bitplanes empacotados com ZIP/DEFLATE; símbolos de índice com zlib e arquivo ZIP sem compressão externa; e bitplanes sem compressão. Isso mede se a distribuição dos índices oferece uma economia adicional. A entropia de ordem zero também é reportada como limite teórico baseado somente nas frequências.

O teste (W X) usa entradas gaussianas sintéticas para conferir como o erro dos pesos afeta uma projeção linear; **não é uma avaliação de linguagem**. Ainda não há kernels customizados para inferência a partir desse formato, nem uma conversão do checkpoint inteiro. O arquivo NPZ é um formato experimental para medir tamanho e validar encoder/decoder, não é diretamente carregável por Transformers.

## Testes locais

```bash
python -m py_compile analyze_weights.py compress_tensor.py
python -m unittest discover -s tests -v
```

O teste unitário cria uma pequena matriz Safetensors sintética e verifica empacotamento/desempacotamento de índices, análise de entropia e round-trip do artefato em cada modo de codificação. Isso valida a mecânica do encoder/decoder, não substitui a execução no Qwen real.

## Próxima etapa experimental

Executar `compress_tensor.py` no checkpoint local e comparar 16, 64, 256 e 512 representantes, com codebooks FP32 e BF16. Se o arquivo real ficar menor e o erro da projeção permanecer aceitável, o próximo teste é repetir por várias camadas e avaliar logits/perplexidade antes de gerar um checkpoint completo.


## Experimento: reconstrução por fórmulas

O script `formula_compress.py` compara **duas fórmulas que ainda usam um índice por peso** com **duas fórmulas que reconstroem blocos inteiros**, além do método K-means existente como referência. Todas as variantes são salvas em NPZ comprimido e medidas pelo tamanho real depois de gravadas.

```bash
# Teste automático em uma matriz intermediária do checkpoint local
python formula_compress.py --model ./Qwen3.5-4B --levels 256 --block-size 256

# Escolher exatamente o tensor testado no experimento anterior
python formula_compress.py --model ./Qwen3.5-4B --tensor-name model.layers.0.self_attn.q_proj.weight --levels 256 --block-size 256 --output-dir resultados_formulas

# Listar candidatos antes de selecionar o tensor
python formula_compress.py --model ./Qwen3.5-4B --list-tensors
```

Métodos incluídos:

| Método | Como reconstrói | Informação guardada |
|---|---|---|
| `baseline_kmeans` | Escolhe o representante aprendido mais próximo | Codebook e um índice por peso |
| `formula_arithmetic_levels` | Gera níveis por uma progressão aritmética: `minimum + índice × step` | Dois parâmetros e um índice por peso |
| `formula_log_companding` | Usa transformação logarítmica assinada e sua inversa | Dois parâmetros e um índice por peso |
| `formula_block_linear` | Ajusta `w(t) = a + bt` para cada bloco contíguo | Dois coeficientes por bloco; sem índices por peso |
| `formula_block_cubic` | Ajusta `w(t) = a + bt + ct² + dt³` para cada bloco | Quatro coeficientes por bloco; sem índices por peso |

Os parâmetros dos polinômios são armazenados em FP16 por padrão; use `--coefficient-dtype fp32` para compará-los em FP32. `--block-size` altera o número de pesos descritos por cada polinômio. O relatório consolidado `*_formula_comparison.json` apresenta o tamanho medido, a redução frente ao tensor de origem, RMSE, erro normalizado, similaridade cosseno e teste sintético `W@X` para os cinco métodos.

**Como interpretar:** as fórmulas escalares economizam a tabela explícita de representantes, mas continuam precisando registrar a escolha de nível de cada peso. Os polinômios eliminam os índices individuais e podem reduzir muito mais o arquivo; se os pesos não apresentarem regularidade local na ordem linear, porém, o erro poderá aumentar bastante. Esse é precisamente o resultado a medir, não uma qualidade presumida. O teste trabalha com um tensor por execução e não valida perplexidade nem qualidade linguística do checkpoint completo.
