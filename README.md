# Análise Comparativa de ResNet-50 e EfficientNet-B0 para Detecção de Melanoma

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[🇺🇸 English](README_EN.md) | [🇧🇷 Português](README.md)

Sistema completo de deep learning para classificação binária de lesões de pele (benignas vs malignas) usando o dataset ISIC 2020, com comparação detalhada entre ResNet-50 e EfficientNet-B0.

## 📋 Índice

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Preparação dos Dados](#preparação-dos-dados)
- [Uso](#uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Resultados](#resultados)

## 🔧 Requisitos

- Python 3.8 ou superior
- CUDA (opcional, mas recomendado para treinamento)
- 8GB+ de RAM
- Espaço em disco: ~5GB (dados + modelos)

## 📦 Instalação

### 1. Clone o repositório ou navegue até o diretório do projeto

```bash
# Se você clonou o repositório:
git clone <url-do-repositorio>
cd melanoma-detection

# Ou se você já tem o projeto, navegue até o diretório:
cd caminho/para/melanoma-detection
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # No macOS/Linux
# ou
venv\Scripts\activate  # No Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

**Nota:** Se `pip` não estiver disponível, use `pip3`:

```bash
pip3 install -r requirements.txt
```

## 📁 Preparação dos Dados

### Estrutura de Diretórios

O projeto espera os dados organizados da seguinte forma:

```
data/
  isic2020/
    benign/
      image1.jpg
      image2.jpg
      ...
    malignant/
      image1.jpg
      image2.jpg
      ...
```

### Opções para Obter os Dados

1. **Dataset ISIC 2020** (oficial):
   - Acesse: https://www.isic-archive.com/
   - Registre-se e baixe o dataset
   - Organize as imagens nas pastas `benign/` e `malignant/`

2. **Dataset alternativo do Kaggle**:
   - Procure por "Skin Cancer: Malignant vs Benign"
   - Baixe e organize na estrutura acima

3. **Dataset de teste pequeno**:
   - Para testes rápidos, você pode usar um subconjunto menor
   - Mantenha a mesma estrutura de pastas

## 🚀 Uso

### 1. Treinar ResNet-50

```bash
python experiments/train.py --config config/resnet50_config.yaml
```

**O que acontece:**
- Carrega e divide o dataset (70% treino, 15% validação, 15% teste)
- Treina o modelo ResNet-50 com early stopping
- Salva checkpoints em `./checkpoints/resnet50/`
- Gera logs no TensorBoard em `./runs/resnet50/`

### 2. Treinar EfficientNet-B0

Primeiro, crie um arquivo de configuração para EfficientNet:

```bash
cp config/resnet50_config.yaml config/efficientnet_config.yaml
```

Edite `config/efficientnet_config.yaml` e altere:
```yaml
model:
  name: efficientnet_b0
```

Depois, treine:

```bash
python experiments/train.py --config config/efficientnet_config.yaml
```

### 3. Comparar Modelos

Após treinar ambos os modelos, execute a comparação completa:

```bash
python experiments/compare.py
```

**O que é gerado:**
- Métricas clínicas comparativas (console)
- Curvas ROC (`results/roc_comparison.png`)
- Gráfico de barras de métricas (`results/metrics_comparison.png`)
- Análise estatística (McNemar, intervalos de confiança)
- Análise de erros (falsos positivos/negativos)
- Benchmark de eficiência computacional

### 4. Análise de Explicabilidade (Grad-CAM)

Gera mapas de atenção visual para comparar como os modelos "veem" as imagens:

```bash
python experiments/analyze_explainability.py \
    --num_samples 100 \
    --resnet_checkpoint ./checkpoints/resnet50/best_model.pth \
    --effnet_checkpoint ./checkpoints/efficientnet_b0/best_model.pth \
    --save_dir ./results/explainability
```

**Resultados:**
- Imagens com Grad-CAM sobreposto
- Página HTML interativa (`results/explainability/index.html`)

## 📊 Visualizar Resultados

### TensorBoard

Para visualizar métricas de treinamento em tempo real:

```bash
tensorboard --logdir ./runs
```

Acesse: http://localhost:6006

### Gráficos Gerados

Os gráficos são salvos em `./results/`:
- `roc_comparison.png` - Curvas ROC comparativas
- `metrics_comparison.png` - Comparação de métricas clínicas

### HTML de Explicabilidade

Abra no navegador:
```
./results/explainability/index.html
```

## 📂 Estrutura do Projeto

```
melanoma-detection/
├── config/                  # Arquivos de configuração YAML
│   └── resnet50_config.yaml
├── data/                   # Dataset e processamento
│   ├── dataset.py          # Carregamento do dataset
│   ├── preprocessing.py    # Transformações e augmentações
│   └── isic2020/           # Dados (benign/, malignant/)
├── models/                 # Arquiteturas de modelos
│   ├── resnet.py
│   ├── efficientnet.py
│   └── model_factory.py
├── training/               # Sistema de treinamento
│   └── trainer.py
├── evaluation/             # Métricas e benchmarks
│   ├── metrics.py          # Métricas clínicas
│   └── efficiency.py        # Benchmark computacional
├── explainability/         # Grad-CAM
│   └── gradcam.py
├── experiments/            # Scripts principais
│   ├── train.py            # Treinamento
│   ├── compare.py          # Comparação completa
│   └── analyze_explainability.py
├── utils/                  # Utilitários
│   ├── config.py            # Gerenciamento de configurações
│   └── reproducibility.py  # Seed e device
├── checkpoints/            # Modelos treinados (gerado)
├── results/                # Resultados e visualizações (gerado)
├── runs/                   # Logs TensorBoard (gerado)
└── requirements.txt        # Dependências
```

## 📈 Métricas Calculadas

O sistema calcula as seguintes métricas:

- **Acurácia**: Taxa de predições corretas
- **Sensibilidade (Recall)**: Taxa de verdadeiros positivos
- **Especificidade**: Taxa de verdadeiros negativos
- **Precisão**: Taxa de predições positivas corretas
- **F1-Score**: Média harmônica de precisão e recall
- **AUC-ROC**: Área sob a curva ROC
- **Cohen's Kappa**: Concordância entre predições e labels
- **MCC**: Matthews Correlation Coefficient

## 🔬 Análise Estatística

A comparação inclui:

- **Intervalos de Confiança (95%)**: Para diferenças entre modelos
- **Teste de McNemar**: Comparação de modelos pareados
- **Análise de Erros**: Falsos positivos/negativos e discordâncias
- **Benchmark de Eficiência**: FLOPs, latência, memória, tamanho

## ⚙️ Configuração

Edite `config/resnet50_config.yaml` para ajustar:

- **Modelo**: `resnet50` ou `efficientnet_b0`
- **Batch size**: Tamanho do lote (padrão: 32)
- **Learning rate**: Taxa de aprendizado (padrão: 0.0001)
- **Épocas**: Número máximo de épocas (padrão: 50)
- **Early stopping**: Patience para parada antecipada (padrão: 10)
- **Augmentações**: Rotação, flip, brilho, contraste, zoom

## 🐛 Solução de Problemas

### Erro: "pip: command not found"

```bash
# Use pip3
pip3 install -r requirements.txt

# Ou instale Python via Homebrew (macOS)
brew install python3
```

### Erro: "CUDA out of memory"

- Reduza o `batch_size` no arquivo de configuração
- Use `device: cpu` se não tiver GPU

### Erro: "Dataset não encontrado"

- Verifique se os dados estão em `./data/isic2020/`
- Confirme a estrutura: `benign/` e `malignant/` dentro de `isic2020/`

### Checkpoints não encontrados

- Execute primeiro o treinamento (`experiments/train.py`)
- Os checkpoints são salvos automaticamente em `./checkpoints/`

## 📝 Exemplo Completo de Execução

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Treinar ResNet-50
python experiments/train.py --config config/resnet50_config.yaml

# 3. Treinar EfficientNet-B0 (após criar config)
python experiments/train.py --config config/efficientnet_config.yaml

# 4. Comparar modelos
python experiments/compare.py

# 5. Análise de explicabilidade
python experiments/analyze_explainability.py --num_samples 50

# 6. Visualizar no TensorBoard
tensorboard --logdir ./runs
```

## 📄 Licença

Este projeto é open source e está licenciado sob a [MIT License](LICENSE).

## 👥 Autores

**Lucas Felipe Cassol Seixas** - [@LSeixas98](https://github.com/LSeixas98)

Projeto desenvolvido para análise comparativa de modelos de deep learning em detecção de melanoma.

## 📚 Referências

- ISIC 2020 Challenge: https://www.isic-archive.com/
- ResNet: He et al. (2016) - Deep Residual Learning
- EfficientNet: Tan & Le (2019) - EfficientNet: Rethinking Model Scaling
- Grad-CAM: Selvaraju et al. (2017) - Grad-CAM: Visual Explanations
