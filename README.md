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


## Experimento profundo: quantização, baixo posto e híbridos

O script `deep_tensor_experiment.py` compara três famílias de representação e mede cada arquivo NPZ real depois de recarregar e decodificar:

1. **K-means com amostras crescentes**: 250 mil, 1 milhão e 2 milhões de pesos, com 64 ou 256 representantes.
2. **Quantização logarítmica**: níveis 128 e 256 com escala característica ajustável (0,25; 0,5; 0,75; 1,0).
3. **Aproximação de baixo posto**: fatores FP16 para postos 64, 128 e 256.
4. **Híbrido**: fatores FP16 de postos 64 ou 128, mais quantização K-means do resíduo usando 16, 32 ou 64 representantes.

Execução no tensor visual já usado nos testes:

```bash
python deep_tensor_experiment.py --model ./Qwen3.5-4B \
  --tensor-name model.visual.merger.linear_fc1.weight \
  --output-dir deep_tensor_results
```

### Codificação de índices comparável

Para K-means, quantização logarítmica e híbridos, o script usa **zlib_symbols por padrão em todos os mapas**, garantindo a mesma codificação de índices durante a comparação. O relatório registra `selected_map_codec`. Para medir também os tamanhos alternativos — bitplanes com ZIP/DEFLATE, símbolos com zlib e bitplanes sem compressão — e selecionar o menor arquivo para cada método, passe `--map-codec auto`; o relatório então inclui `candidate_codec_sizes_bytes`.

```bash
# Comparar todos os codificadores e escolher o menor arquivo por método
python deep_tensor_experiment.py --model ./Qwen3.5-4B \
  --tensor-name model.visual.merger.linear_fc1.weight \
  --map-codec auto --output-dir deep_tensor_auto_codec
```

Ajustes avançados podem ser feitos por `--kmeans-samples`, `--kmeans-groups`, `--log-levels`, `--log-scales`, `--ranks`, `--hybrid-ranks` e `--residual-groups`. Use `--timing-repeats` para controlar as repetições do benchmark exploratório de projeção fatorada.

O relatório `deep_comparison.json` contém as configurações, tamanho real do arquivo, codificador escolhido, RMSE normalizado, similaridade cosseno, erro relativo W@X e tempo de decodificação offline. `deep_comparison.csv` facilita comparar os métodos. Os arquivos NPZ individuais e relatórios detalhados ficam no diretório de saída.

Para candidatos de baixo posto, o script também compara `U @ (V.T @ X)` com `W @ X` denso usando NumPy FP32 na CPU, sem reconstruir a matriz (W) durante a multiplicação fatorada. Esse é um benchmark de operação matricial, não tokens por segundo do modelo e não uma previsão de velocidade na GPU.

**Limitações:** randomized SVD é uma aproximação; mais `--oversample` e `--power-iterations` podem aumentar a precisão com custo de tempo. O teste W@X usa entrada sintética e não substitui a avaliação de logits, ativações reais ou perplexidade do modelo inteiro.


## Codec de checkpoint completo

`full_model_codec.py` aplica a fórmula logarítmica ao **checkpoint inteiro**, em vez de escolher uma única matriz. Ele processa todos os tensores Safetensors em blocos para limitar memória, usa 256 níveis e multiplicador de escala 0,75 por padrão, e estima uma escala por tensor a partir de uma amostra determinística. Tensores pequenos (até 256 elementos) e tensores não flutuantes são preservados sem perda.

### Comprimir todos os pesos

```bash
python full_model_codec.py compress \
  --model ./Qwen3.5-4B \
  --output ./qwen3.5-4b-log256-s075.rccomp
```

O artefato `.rccomp` contém todos os tensores e um manifesto, num contêiner ZIP64. Cada fluxo de índices é comprimido com zlib e armazenado sem uma segunda camada de compressão. Após a gravação, o programa percorre os payloads para verificar seus comprimentos e hashes SHA-256. Use `--skip-verify` para pular essa leitura adicional caso precise reduzir o tempo de execução; nesse modo não haverá verificação integral dos payloads gravados.

O relatório `qwen3.5-4b-log256-s075.rccomp.report.json` informa tamanho real do arquivo, redução frente aos bytes originais dos tensores, erro numérico agregado sobre todos os pesos flutuantes, contagem por tipo de representação e métricas individuais por tensor. O tamanho é medido depois da gravação, não estimado.

### Reconstruir um checkpoint Safetensors para avaliar o modelo

```bash
python full_model_codec.py decode \
  --archive ./qwen3.5-4b-log256-s075.rccomp \
  --output-dir ./Qwen3.5-4B-log256-s075 \
  --source-model ./Qwen3.5-4B
```

O decodificador recria os tensores no dtype original e divide a saída em shards Safetensors. Ele também copia os arquivos auxiliares do modelo (configuração, tokenizer e processador) a partir da pasta de origem. O checkpoint reconstruído pode então ser carregado pelo Transformers para verificar geração, logits ou perplexidade. Reserve espaço em disco para o arquivo comprimido e para o checkpoint reconstruído; a decodificação não altera o original.

### Limites desta etapa

- O formato `.rccomp` é um formato de pesquisa próprio e não pode ser carregado diretamente pelo Transformers.
- A métrica agregada dos pesos não prova que a qualidade linguística foi preservada; a confirmação exige inferência ou perplexidade no checkpoint reconstruído.
- A inferência ainda usa os pesos BF16 reconstruídos. Ganho de velocidade durante a geração exigirá um runtime/kernel que opere diretamente sobre o formato comprimido.
- A escala por tensor é estimada por amostragem; a escala e o limite de transformação são registrados no manifesto para permitir reconstrução reproduzível.


## Teste mínimo de resposta do checkpoint reconstruído

Para confirmar apenas que o modelo carrega e responde, `smoke_test_model.py` faz uma única geração curta. Não atualize o PyTorch isoladamente: `torch`, `torchvision` e `torchaudio` precisam ter versões compatíveis. Para uma GTX 1050 Ti (arquitetura Pascal), prefira a distribuição CUDA 12.6; as distribuições CUDA 13.x mais recentes deixaram de incluir suporte a algumas arquiteturas antigas.

No Windows/PowerShell, crie um ambiente isolado para não alterar as dependências dos outros projetos:

```powershell
py -3.11 -m venv .venv-smoke
.\.venv-smoke\Scripts\python.exe -m pip install --upgrade pip
.\.venv-smoke\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-smoke\Scripts\python.exe -m pip install -U transformers accelerate huggingface_hub
.\.venv-smoke\Scripts\python.exe -m pip check
.\.venv-smoke\Scripts\python.exe smoke_test_model.py --model ".\Qwen3.5-4B-log256-s075"
```

O script usa FP16 por padrão e `device_map="auto"`, que pode distribuir camadas entre a GPU e a CPU. O checkpoint em disco permanece no dtype reconstruído. Se a importação de um componente do Transformers falhar, o script exibe as versões instaladas e o traceback completo. O teste limita a geração a 24 tokens e desativa o modo de raciocínio do Qwen3.5 para evitar gastar o orçamento curto em uma cadeia de pensamento. Se o processador/tokenizer reconstruído não tiver `chat_template`, ele usa os tokens de conversa do Qwen e fecha o bloco `<think>` vazio para solicitar uma resposta direta; também considera que parte dos pesos pode estar descarregada em CPU/disco.

O teste termina com `[PASS]` se o modelo produzir texto não vazio. Isso comprova apenas carregamento e geração básica, não equivalência de qualidade com o modelo original.


## Fórmula global para reconstruir uma matriz de camada

O script `formula_fit_layer.py` extrai **todos os pesos de uma única matriz 2D real** para `weights_original.npy` e ajusta aproximações por uma fórmula trigonométrica global (série de Fourier 2D). Em vez de guardar cada peso, cada variante guarda apenas coeficientes de frequência baixa; os pesos aproximados são gerados pela fórmula inversa. Os coeficientes são armazenados como pares FP16 e o script mede o tamanho real do arquivo salvo.

```powershell
# Listar matrizes/tensores do checkpoint local
python formula_fit_layer.py --model ".\Qwen3.5-4B" --list-tensors

# Escolher automaticamente uma matriz 2D de tamanho intermediário
python formula_fit_layer.py --model ".\Qwen3.5-4B" --output-dir formula_fit_layer_results

# Ou especificar o tensor exato exibido por --list-tensors
python formula_fit_layer.py --model ".\Qwen3.5-4B" --tensor-name "model.visual.merger.linear_fc1.weight"

# Reduzir ou aumentar o orçamento de coeficientes
python formula_fit_layer.py --model ".\Qwen3.5-4B" --frequencies 8,16,32,64,128,256
```

A fórmula usada é uma soma de senos/cossenos em função das coordenadas da matriz. `--frequencies` define o maior índice de frequência preservado em ambas as dimensões: valores menores armazenam menos coeficientes, mas descartam mais variação. O script testa cada variante usando todos os pesos da matriz, recarrega os coeficientes salvos, reconstrói os valores e reporta RMSE, RMSE normalizado pelo desvio-padrão, erro L2 relativo, similaridade cosseno, tamanho real do artefato e bits efetivos por peso. Resultados em `formula_fit_report.json`.

**Interpretação:** isso testa uma fórmula que gera a matriz sem armazenar um mapa de índices por peso. Se os pesos tiverem pouca estrutura espacial, uma fórmula compacta poderá ter erro próximo ao desvio-padrão original — indicando que ela não conseguiu preservar a matriz. O experimento mede aproximação numérica de pesos; não presume qualidade de geração. Os coeficientes de Fourier não são índices de representantes, e o custo de calcular a fórmula durante a inferência ainda não é avaliado.


## Fórmula não linear aprendida para uma camada

`nonlinear_formula_layer.py` testa uma representação diferente da série de Fourier fixa: uma função não linear pequena, com parâmetros aprendidos a partir de **todos os pesos de uma matriz real**. Cada peso é calculado a partir de um código compacto compartilhado pela sua linha, outro pela sua coluna e uma MLP compartilhada. O artefato não guarda índices nem correções individuais por peso.

A fórmula usada é:

```text
W_hat[r,c] = média + desvio * (
    MLP(código_linha[r], código_coluna[c])
    + viés_linha[r] + viés_coluna[c]
)
```

Execução no PowerShell, usando o checkpoint local:

```powershell
python nonlinear_formula_layer.py --model ".\Qwen3.5-4B" --output-dir nonlinear_formula_layer_results
```

A seleção automática escolhe uma matriz 2D intermediária de até 20 milhões de pesos; para usar uma matriz específica, consulte `--list-tensors` e passe `--tensor-name`. As opções `--embedding-dim`, `--hidden-dim` e `--epochs` controlam o tamanho e o treinamento da fórmula. O padrão faz cinco passadas por todas as linhas.

Resultados:
- `nonlinear_weight_formula.npz`: somente os parâmetros aprendidos e metadados da fórmula, armazenados em FP16 e comprimidos.
- `nonlinear_formula_report.json`: tamanho real do artefato, bits efetivos por peso, redução no armazenamento, RMSE normalizado, erro L2 relativo e similaridade cosseno.
- `weights_nonlinear_reconstructed.npy`: matriz prevista, somente se passar `--save-reconstructed` (gera um arquivo grande para análise).

O número de parâmetros deve ser muito menor que o de pesos para haver compressão. Como a função precisa aprender a distribuição real da matriz, o tempo de ajuste pode aumentar. Uma redução grande de tamanho com erro normalizado próximo de 1 significa que a função não aprendeu informação suficiente; isso não deve ser interpretado como compressão bem-sucedida do modelo. O resultado mede primeiro a aproximação numérica de uma matriz, não a qualidade linguística nem a velocidade de inferência.


## Experimento: mapa de representantes preditivo

`predictive_map_experiment.py` testa se um decodificador pequeno consegue explorar padrões no mapa de índices de uma matriz real. O tensor é quantizado para até 16 representantes (4 bits por código), depois o script compara o mapa direto com resíduos preditivos usando uma tabela condicional minúscula. O preditor usa o código imediatamente acima de cada posição; também testa a matriz transposta, que corresponde a prever pelo código anterior na outra dimensão.

O decoder trabalha uma linha por vez e vetoriza todas as colunas daquela linha. O teste informa tanto o tamanho real em disco quanto a velocidade de decodificação, porque uma economia de espaço que torne a reconstrução lenta não atende ao objetivo.

```powershell
# Usar a mesma matriz visual do experimento anterior
python predictive_map_experiment.py --model ".\Qwen3.5-4B" --tensor-name "model.visual.merger.linear_fc1.weight" --output-dir predictive_map_results

# Listar tensores para escolher outra matriz
python predictive_map_experiment.py --model ".\Qwen3.5-4B" --list-tensors
```

Arquivos de saída:
- `predictive_representative_map.npz`: representantes FP16, tabela preditiva e payload comprimido do mapa residual.
- `predictive_map_report.json`: tamanho do mapa direto e do mapa preditivo, entropia dos índices/resíduos, acerto do preditor, erro da reconstrução, e velocidade em milhões de índices por segundo.

O mapa preditivo tem de ser decodificado **exatamente** para recuperar os índices; a aproximação numérica resulta apenas dos 16 representantes. Esse é um primeiro teste de referência, não uma micro-LLM: se a estrutura local já não reduzir o mapa de modo significativo ou ficar mais lenta, isso mede o obstáculo que um decodificador neural mais complexo teria de superar. O tamanho de uma tabela preditiva aprendida por matriz é contabilizado no artefato salvo.


## Análise de neurônios MLP potencialmente redundantes

`analyze_neuron_similarity.py` analisa uma camada MLP real procurando neurônios cujos três vetores de pesos sejam muito parecidos:
- a linha de `gate_proj`;
- a linha de `up_proj`;
- a coluna correspondente de `down_proj`.

O teste compara todos os pares por cosseno quando CUDA está disponível. Também verifica se as escalas são compatíveis e aceita que `up_proj` e `down_proj` tenham sinais invertidos ao mesmo tempo, pois esse padrão pode conservar a contribuição multiplicativa do ramo. Na CPU, por padrão usa uma amostra de 1.024 neurônios para evitar uma comparação quadrática demasiado lenta.

```powershell
# Usar o Python do ambiente isolado do teste anterior
.\.venv-smoke\Scripts\python.exe analyze_neuron_similarity.py --model ".\Qwen3.5-4B" --layer-index 0 --output-dir neuron_similarity_results

# Listar as camadas encontradas
.\.venv-smoke\Scripts\python.exe analyze_neuron_similarity.py --model ".\Qwen3.5-4B" --list-layers

# Na GPU, aumentar o bloco de comparação se houver memória disponível
.\.venv-smoke\Scripts\python.exe analyze_neuron_similarity.py --model ".\Qwen3.5-4B" --layer-index 0 --batch-rows 128
```

O relatório `neuron_similarity_report.json` conta pares nos quais **todos os três vetores** ultrapassam os limiares de similaridade 0,90, 0,95, 0,98 e 0,99, respeitando também os filtros de escala/sinal. Salva ainda os melhores pares candidatos e os cossenos individuais.

**Limitação importante:** este é um primeiro filtro baseado nos pesos, não uma prova de equivalência funcional. Para confirmar que dois neurônios são de fato substituíveis, seria necessário comparar suas ativações e contribuições usando entradas reais. O script não poda nem altera o checkpoint.


## Análise funcional de neurônios por ativações reais

`analyze_neuron_activations.py` complementa a comparação de pesos: carrega o checkpoint com `device_map="auto"`, executa o forward somente até a MLP escolhida e interrompe ali, sem calcular as camadas seguintes. Captura as ativações de `gate_proj` e `up_proj` e calcula a ativação gated de cada neurônio. Então procura pares com traços de ativação correlacionados e estima o erro de saída se a contribuição de um neurônio for fundida no outro, ajustando a coluna correspondente de `down_proj`.

```powershell
# Analisar a camada 0 com um único texto de calibração
.\.venv-smoke\Scripts\python.exe analyze_neuron_activations.py --model ".\Qwen3.5-4B" --layer-index 0 --max-tokens 256 --output-dir neuron_activation_results

# Usar texto local para melhorar a amostra de calibração, sem executar geração
.\.venv-smoke\Scripts\python.exe analyze_neuron_activations.py --model ".\Qwen3.5-4B" --layer-index 0 --text-file ".\calibration_multilingual.txt" --max-tokens 512
```

O arquivo `neuron_activation_report.json` registra quantos pares têm correlação absoluta de ativação acima de 0,90, 0,95, 0,98 e 0,99, além dos melhores pares. Para cada candidato, estima o erro na contribuição combinada dos dois neurônios e uma estimativa do erro relativo perante a saída total da MLP se um fosse absorvido no outro.

Embora as camadas posteriores não sejam executadas, `from_pretrained` ainda inicializa/encaminha o checkpoint inteiro e pode descarregar pesos em CPU/disco; portanto, o carregamento inicial ainda custa tempo. A interrupção reduz o cálculo do forward. Essa é uma aproximação funcional melhor que comparar apenas os pesos, mas os resultados dependem do texto usado. Um único texto serve para filtrar candidatos; antes de podar, seria necessário confirmar os melhores pares em mais entradas. O script apenas analisa e não modifica o checkpoint.

## Mapa esparso de correções individuais para Q4_0

O script `q4_correction_map.py` compara cada peso de origem com a reconstrução Q4_0 e gera mapas esparsos independentes para os limites absolutos `1.0`, `0.8`, `0.6`, `0.5`, `0.4`, `0.3`, `0.2`, `0.1`, `0.08`, `0.05`, `0.03` e `0.01`. Para cada peso acima do limite, armazena apenas o índice global delta-coded e o resíduo FP16 (`original - Q4`). A correção aplicada é `Q4 + resíduo`; o script mede também o erro restante causado pelo armazenamento do resíduo em FP16.

```powershell
.\.venv-smoke\Scripts\python.exe q4_correction_map.py `
  --model ".\Qwen3.5-4B" `
  --thresholds 1.0,0.8,0.6,0.5,0.4,0.3,0.2,0.1,0.08,0.05,0.03,0.01 `
  --output-dir q4_correction_map_results
```

Para cada limite, cria `correction_map_gt_*.bin`; `report.json` informa número de correções, blocos/tensores atingidos, tamanho real do mapa, maior erro entre os pesos corrigidos, erro máximo global após aplicar o mapa, RMSE global e contagem de pesos que ainda ultrapassam o limite. `tensor_manifest.json` mapeia os índices globais para tensores e offsets, inclusive os que foram excluídos da simulação. Os bytes do mapa são medidos no arquivo real e não incluem o modelo Q4 base nem o manifesto.

**Interpretação:** este é um teste de tamanho e erro numérico com Q4_0 simulado diretamente a partir do checkpoint, não uma quantização do checkpoint para um arquivo GGUF nem um teste de perplexidade/qualidade de geração. Um resíduo FP16 é uma correção aproximada, e os mapas são dados experimentais; ainda não há um carregador de inferência que aplique esses mapas a um modelo Q4 externo.

## Segundo teste: pesos com erro absoluto maior que um limite

`q4_weight_deviation.py` agora aceita `--error-threshold` (padrão `1.0`) e salva **todos** os pesos em que `abs(Q4_reconstruído - BF16_original) > limite`. A comparação usa o erro absoluto individual, não médias, RMSE ou desvio-padrão. O valor 1.0 é aplicado nas mesmas unidades numéricas dos pesos.

```powershell
.\.venv-smoke\Scripts\python.exe q4_weight_deviation.py `
  --model ".\Qwen3.5-4B" `
  --error-threshold 1.0 `
  --output-dir q4_error_gt_1_results
```

O comando produz:
- `weights_above_threshold.csv`: **cada peso** com erro absoluto estritamente maior que 1.0, mostrando índice, coordenadas, valor de origem, valor Q4 reconstruído e diferença.
- `blocks_above_threshold.csv`: cada bloco Q4_0 com pelo menos um peso acima de 1.0, quantos pesos ultrapassaram o limite e quais posições locais foram afetadas.
- `tensor_threshold_counts.csv`: quantos pesos e blocos ultrapassam o limite em cada tensor, inclusive tensores com zero ocorrências.
- `report.json`: contagem global de pesos, blocos e tensores que ultrapassaram o limite.

Para mudar o limite, use por exemplo `--error-threshold 0.5` ou `--error-threshold 2.0`. O teste ainda usa a simulação Q4_0 existente e só processa tensores alinháveis a blocos de 32 sem cruzar as fronteiras das linhas; veja `skipped_tensors.json` para a lista excluída. Os resultados permitem selecionar posições para um mapa de correções, mas um erro numérico alto não prova sozinho que aquele peso cause grande impacto na saída do modelo.

## Mapa dos desvios individuais BF16 -> Q4_0

O script `q4_weight_deviation.py` foi criado para localizar os pesos que mais mudam na quantização, sem classificar a perda por média, RMSE ou desvio-padrão. Ele compara cada valor do Safetensors original com o valor reconstruído após Q4_0 e registra a posição exata, o delta assinado e o erro absoluto.

No PowerShell, analise todos os tensores elegíveis:

```powershell
.\.venv-smoke\Scripts\python.exe q4_weight_deviation.py --model ".\Qwen3.5-4B" --output-dir q4_weight_deviation_results
```

Saídas principais:

- `top_weight_deviations.csv`: os pesos individuais com maior diferença absoluta em todo o modelo, incluindo tensor, coordenadas, valor BF16, valor Q4 reconstruído e delta.
- `top_block_deviations.csv`: blocos de 32 pesos ordenados pelo pior peso de cada bloco.
- `worst_weight_per_tensor.csv`: o peso que mais se desviou dentro de cada tensor.
- `report.json` e `skipped_tensors.json`: escopo e tensores que não foram tratados como Q4_0 porque o tamanho não é múltiplo de 32.

Para obter o erro de **cada peso** em uma matriz específica (o CSV pode ficar grande):

```powershell
.\.venv-smoke\Scripts\python.exe q4_weight_deviation.py --model ".\Qwen3.5-4B" --tensor-name "model.language_model.layers.0.mlp.gate_proj.weight" --dump-all-weights --output-dir gate_proj_q4_deviation
```

Esse modo gera também `all_weight_errors.csv`, com uma linha por peso e seu índice/posição. `--top-k 2000` aumenta a quantidade de piores pesos e blocos listados. Use o nome exato do tensor retornado pelo script existente `compare_representatives_vs_q4.py --list-tensors`.

**Importante:** este teste quantiza os valores do Safetensors de origem com uma implementação Q4_0 em blocos e reconstrói os valores armazenados (inclusive a escala FP16); confira `source_dtype` nos CSVs para confirmar se a origem é `bfloat16`. Ele não lê nem decodifica um arquivo GGUF Q4 externo. As métricas de ranking são por peso/bloco, sem médias globais.

## Comparação direta: representantes globais, por linha e Q4_0

`compare_representatives_vs_q4.py` compara três representações **da mesma matriz**. Por padrão usa `model.language_model.layers.0.mlp.gate_proj.weight`:

- **Representantes globais:** 16 valores FP16 aprendidos da distribuição da matriz inteira; cada peso usa um índice de 4 bits.
- **Representantes por linha:** 16 valores FP16 independentes para cada linha da matriz, também com índices de 4 bits. Essa é a nova variante local, que adiciona custo para o codebook mas adapta os níveis à distribuição de cada linha.
- **Q4_0:** formato de referência do GGML com blocos de 32 pesos, 16 bytes de índices compactados e uma escala FP16 por bloco — 18 bytes por 32 pesos (4,5 bits/peso). A disposição segue a [documentação do llama.cpp](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes).

Execute no PowerShell com o checkpoint local:

```powershell
.\.venv-smoke\Scripts\python.exe compare_representatives_vs_q4.py --model ".\Qwen3.5-4B" --output-dir representatives_vs_q4_results
```

Para listar as matrizes ou selecionar outra:

```powershell
.\.venv-smoke\Scripts\python.exe compare_representatives_vs_q4.py --model ".\Qwen3.5-4B" --list-tensors
.\.venv-smoke\Scripts\python.exe compare_representatives_vs_q4.py --model ".\Qwen3.5-4B" --tensor-name "model.language_model.layers.0.mlp.gate_proj.weight"
```

O relatório `representatives_vs_q4_report.json` registra tamanho do mapa e das tabelas, bits efetivos por peso, RMSE normalizado pelo desvio-padrão, erro L2 relativo, erro absoluto médio e similaridade cosseno. Os métodos são reconstruídos a partir dos índices empacotados e dos codebooks FP16, para que as métricas reflitam os valores efetivamente armazenados.

Na matriz padrão de 9.216 linhas, com 16 representantes FP16 por linha, o codebook ocupa 294.912 bytes (cerca de 0,295 MB) além do mapa de 4 bits por peso. O programa calcula o custo real a partir das dimensões da matriz, em vez de assumir esse tamanho para todas as matrizes.

**Escopo:** o experimento compara erro numérico dos pesos, não perplexidade, qualidade de geração ou velocidade de inferência. Q4_0 é uma base Q4 simples em blocos; não deve ser confundido com Q4_K_M. O payload contabilizado não inclui cabeçalhos de um contêiner como NPZ.


## Comparação real: BF16 original vs. checkpoint bitsandbytes NF4

`compare_bnb_weights.py` compara o checkpoint BNB NF4 já salvo com seus pesos BF16 de origem. Diferentemente dos experimentos Q4_0 anteriores, o script **não simula uma nova quantização**: lê os códigos NF4 e seu `QuantState` diretamente dos Safetensors, desquantiza um tensor por vez e compara os valores reconstruídos com a origem.

O destino padrão é `techwithsergiu/Qwen3.5-text-4B-bnb-4bit`, cuja origem direta é `techwithsergiu/Qwen3.5-text-4B`. Se você já tem o Qwen multimodal original em `.\Qwen3.5-4B`, pode usá-lo como fonte: o script tenta mapear automaticamente `model.layers.*` do modelo somente de texto para `model.language_model.layers.*` da origem multimodal. Os tensores visuais removidos aparecem como tensores presentes apenas na origem e não entram na comparação.

Instale bitsandbytes no ambiente que já contém PyTorch:

```powershell
.\.venv-smoke\Scripts\python.exe -m pip install bitsandbytes
```

Execute a comparação no modelo original local e no NF4 publicado:

```powershell
.\.venv-smoke\Scripts\python.exe compare_bnb_weights.py `
  --source-model ".\Qwen3.5-4B" `
  --quantized-model "techwithsergiu/Qwen3.5-text-4B-bnb-4bit" `
  --device cuda `
  --thresholds 1,0.5,0.1,0.05,0.01 `
  --output-dir bnb_weight_validation
```

Para gerar o mapa de correções reais durante a mesma varredura, acrescente `--export-corrections --correction-threshold 0.01`:

```powershell
.\.venv-smoke\Scripts\python.exe compare_bnb_weights.py `
  --source-model ".\Qwen3.5-4B" `
  --quantized-model "techwithsergiu/Qwen3.5-text-4B-bnb-4bit" `
  --device cuda `
  --export-corrections `
  --correction-threshold 0.01 `
  --output-dir bnb_weight_validation
```

O primeiro uso baixa o checkpoint NF4 de aproximadamente 3,12 GB se ele ainda não estiver no cache. Para usar os arquivos BF16 text-only exatos que deram origem à quantização, troque `--source-model` por `techwithsergiu/Qwen3.5-text-4B`; isso pode exigir baixar vários GB adicionais. Também é possível apontar `--quantized-model` a um diretório local já baixado.

Saídas:
- `report.json`: erro global de todos os tensores pareados e métricas separadas para pesos NF4 e tensores que permaneceram em maior precisão.
- `tensor_comparison.csv`: MAE, RMSE, erro máximo, similaridade cosseno e contagem acima de cada limite para cada tensor.
- `top_weight_deviations.csv`: os maiores desvios individuais, com tensor, índice, coordenadas, peso de origem, peso NF4 reconstruído e delta.
- Quando `--export-corrections` é usado, `correction_map_gt_0p01.bin` guarda índices delta-coded locais a cada tensor e resíduos FP16; `correction_map_manifest.json` indica offsets, dimensões e contagens para decodificar cada seção.

O mapa auxiliar é apenas exportado: ele **não modifica o checkpoint nem é aplicado automaticamente pelo Transformers**. A aplicação durante a inferência ainda exige um módulo que interprete o manifesto, decodifique as entradas e some a contribuição dos resíduos nas camadas correspondentes.

**Importante:** o script usa a desquantização do bitsandbytes sobre os dados realmente armazenados; não quantiza de novo os pesos de origem. Ele mede diferenças numéricas dos pesos, não perplexidade nem qualidade de geração. Se a desquantização NF4 falhar no backend CUDA instalado, o script interrompe com o nome do tensor em vez de substituir silenciosamente o resultado por uma simulação.


## Teste de inferência com o mapa esparso NF4

`apply_bnb_corrections.py` carrega o checkpoint BNB NF4 normalmente e instala forward hooks PyTorch nas camadas `Linear4bit` listadas em `correction_map_manifest.json`. Em cada camada, calcula a contribuição dos resíduos esparsos (`E @ x`) e soma essa contribuição à saída quantizada. O checkpoint original não é alterado.

O teste roda uma passagem sem correções e outra com correções, compara os logits do último token e gera texto com ambas as configurações. É um protótipo para validar mapeamento das camadas, efeito numérico e custo de tempo — não uma implementação otimizada para produção.

Execute usando os arquivos criados pelo comando `compare_bnb_weights.py --export-corrections`:

```powershell
.\.venv-smoke\Scripts\python.exe apply_bnb_corrections.py `
  --model "techwithsergiu/Qwen3.5-text-4B-bnb-4bit" `
  --map-file ".\bnb_weight_validation\correction_map_gt_0p01.bin" `
  --manifest ".\bnb_weight_validation\correction_map_manifest.json" `
  --prompt "Explique brevemente por que o céu parece azul." `
  --max-new-tokens 48 `
  --device-map gpu `
  --compute-dtype float16 `
  --output ".\bnb_weight_validation\correction_runtime_test.json"
```

O script valida cada tensor do manifesto contra um módulo `Linear4bit` de dimensões compatíveis. Ele também converte automaticamente o prefixo `model.language_model.layers.*` do manifesto em `model.layers.*` quando esse for o nome exposto pelo `AutoModelForCausalLM` text-only, registrando quantos aliases foram resolvidos. Se alguma correção continuar sem correspondência, ele interrompe em vez de ignorá-la silenciosamente. Por padrão, `--device-map gpu` força o modelo inteiro para CUDA 0 para evitar o erro do bitsandbytes 4-bit quando `device_map="auto"` envia módulos para CPU/disco. Isso pode causar CUDA OOM se a VRAM livre não for suficiente; nesse caso, o modelo não será automaticamente dividido entre CPU e GPU. O carregador também modifica a configuração NF4 em memória para usar o dtype escolhido, sem passar uma segunda `quantization_config` que seria ignorada pelo Transformers. O JSON de saída inclui quantidade de hooks, diferença máxima/RMSE dos logits, mudança do próximo token, textos gerados sem/com mapa e tempos de geração.

**Limitação:** o hook implementa a contribuição esparsa por gathers e `scatter_add`; adiciona trabalho em cada camada com correções e pode reduzir a velocidade, especialmente durante geração token a token. Primeiro valide a correção e o resultado numérico; otimização de desempenho é uma etapa posterior. O teste usa um único prompt e não substitui avaliação de perplexidade ou de qualidade em um conjunto de tarefas.
