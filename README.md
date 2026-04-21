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

### 🖥️ Interface Gráfica (Recomendado para Iniciantes)

**Nova funcionalidade!** Interface web completa para treinar, fazer predições e analisar resultados de forma intuitiva.

#### Iniciar a Interface

**Windows:**
```bash
python app.py
# ou clique duas vezes em run_gui.bat
```

**Linux/Mac:**
```bash
python3 app.py
# ou execute: ./run_gui.sh
```

A interface será aberta automaticamente em: **http://localhost:7860**

#### Funcionalidades Principais

**🏋️ Treinamento**
- Selecione o modelo (ResNet-50 ou EfficientNet-B0)
- Configure hiperparâmetros (épocas, batch size, learning rate)
- Acompanhe o progresso em tempo real com logs ao vivo
- Checkpoints salvos automaticamente em `./checkpoints/`

**🔍 Predição Individual**
- Carregue modelos treinados com um clique
- Faça upload de imagens de lesões cutâneas
- Visualize probabilidades e predições
- **Grad-CAM**: Mapas de calor mostrando regiões importantes para a decisão
- Compare predições de ambos os modelos lado a lado

**📁 Predição em Lote**
- Processe múltiplas imagens simultaneamente
- Exportação de resultados em CSV
- Visualização em tabela com probabilidades

**⚖️ Comparação de Modelos**
- Compare ResNet-50 vs EfficientNet-B0
- Métricas clínicas comparativas (Acurácia, Sensibilidade, Especificidade, AUC-ROC, F1-Score)
- Gráficos comparativos interativos

**📊 Histórico de Treinamentos**
- Visualize todos os treinamentos realizados
- Curvas de aprendizado (loss, AUC-ROC, learning rate)
- Métricas de teste de cada modelo

#### Como Usar - Fluxo Rápido

1. **Treinar um modelo:**
   - Aba "🏋️ Treinamento" → Selecione modelo → Configure parâmetros → "▶️ Iniciar Treinamento"
   - Aguarde conclusão (progresso em tempo real)

2. **Fazer predições:**
   - Aba "🔍 Predição Individual" → Carregue os modelos → Upload da imagem → "🔮 Fazer Predição"
   - Marque "Mostrar Grad-CAM" para visualização de explicabilidade

3. **Comparar resultados:**
   - Aba "⚖️ Comparação" → "🔄 Executar Comparação"
   - Analise métricas e identifique o melhor modelo

#### Dicas Importantes

- **Sempre carregue os modelos** antes de fazer predições (botões "Carregar ResNet-50" e "Carregar EfficientNet-B0")
- **Use Grad-CAM** para validar se o modelo está focando em regiões corretas da lesão
- **Compare ambos os modelos** para maior confiança nas predições
- **Sensibilidade é crucial** para melanoma (capacidade de detectar casos malignos)

#### Compartilhar na Rede Local

Para acessar de outros dispositivos na mesma rede:

1. Identifique seu IP local:
   ```bash
   ipconfig      # Windows
   ifconfig      # Linux/Mac
   ```

2. Acesse de outro dispositivo:
   ```
   http://<SEU_IP>:7860
   ```

#### Solução de Problemas Comuns

- **"Modelo não carregado"**: Treine o modelo primeiro ou verifique o caminho do checkpoint
- **"CUDA out of memory"**: Reduza o batch size ou use CPU (detectado automaticamente)
- **Logs não atualizam**: Clique em "🔄 Atualizar Logs" manualmente
- **Interface travada**: Aguarde conclusão do processo ou reinicie o servidor (Ctrl+C)

#### Monitoramento Avançado

Para análise detalhada durante treinamento, use TensorBoard em paralelo:
```bash
tensorboard --logdir=./runs
```
Acesse em: http://localhost:6006

📖 **Documentação completa da GUI**: [GUI_README.md](GUI_README.md) | **Guia rápido**: [QUICKSTART.md](QUICKSTART.md)

---

### 🎯 Linha de Comando (Avançado)

Para executar todo o pipeline automaticamente (treinar ambos os modelos, comparar e gerar relatório):

```bash
python experiments/main.py
```

**O que acontece:**
1. Treina ResNet-50
2. Treina EfficientNet-B0
3. Compara ambos os modelos
4. Salva resultados em `./results/results.json`

Depois, gere o relatório completo:

```bash
python experiments/generate_report.py
```

**Resultados:**
- Relatório Markdown: `./results/relatorio_completo.md`
- Relatório HTML: `./results/relatorio_completo.html` (se markdown instalado)
- Todos os checkpoints, logs e visualizações

---

### Executar Passo a Passo (Manual)

Se preferir executar cada etapa separadamente:

#### 1. Treinar ResNet-50

```bash
python experiments/train_resnet.py
```

**O que acontece:**
- Carrega e divide o dataset (70% treino, 15% validação, 15% teste)
- Treina o modelo ResNet-50 com early stopping
- Salva checkpoints em `./checkpoints/resnet50/`
- Gera logs no TensorBoard em `./runs/resnet50/`

#### 2. Treinar EfficientNet-B0

```bash
python experiments/train_efficientnet.py
```

**O que acontece:**
- Carrega e divide o dataset (70% treino, 15% validação, 15% teste)
- Treina o modelo EfficientNet-B0 com early stopping
- Salva checkpoints em `./checkpoints/efficientnet_b0/`
- Gera logs no TensorBoard em `./runs/efficientnet_b0/`

#### 3. Comparar Modelos

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

#### 4. Análise de Explicabilidade (Grad-CAM)

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
├── app.py                  # Interface gráfica principal (NOVO!)
├── gui/                    # Módulos da interface gráfica (NOVO!)
│   ├── training_interface.py    # Interface de treinamento
│   ├── prediction_interface.py  # Interface de predição
│   └── comparison_interface.py  # Comparação e histórico
├── data/                   # Dataset e processamento
│   ├── dataset.py          # Carregamento do dataset
│   ├── preprocessing.py    # Transformações e augmentações
│   └── isic2020/           # Dados (benign/, malignant/)
├── models/                 # Arquiteturas de modelos
│   ├── resnet.py
│   └── efficientnet.py
├── training/               # Sistema de treinamento
│   └── trainer.py
├── evaluation/             # Métricas e benchmarks
│   ├── metrics.py          # Métricas clínicas
│   └── efficiency.py        # Benchmark computacional
├── explainability/         # Grad-CAM
│   └── gradcam.py
├── experiments/            # Scripts principais (linha de comando)
│   ├── main.py             # Script principal (executa tudo)
│   ├── generate_report.py  # Gera relatório completo
│   ├── train_resnet.py     # Treinar ResNet-50
│   ├── train_efficientnet.py # Treinar EfficientNet-B0
│   ├── compare.py          # Comparação completa
│   └── analyze_explainability.py
├── utils/                  # Utilitários
│   └── reproducibility.py  # Seed e device
├── checkpoints/            # Modelos treinados (gerado)
├── results/                # Resultados e visualizações (gerado)
├── runs/                   # Logs TensorBoard (gerado)
├── requirements.txt        # Dependências
└── GUI_README.md           # Documentação da interface gráfica (NOVO!)
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

Os scripts de treinamento (`train_resnet.py` e `train_efficientnet.py`) têm configurações simples e diretas. Para ajustar parâmetros, edite diretamente os scripts:

- **Batch size**: Tamanho do lote (padrão: 32)
- **Learning rate**: Taxa de aprendizado (padrão: 0.0001)
- **Épocas**: Número máximo de épocas (padrão: 50)
- **Early stopping**: Patience para parada antecipada (padrão: 10)
- **Augmentações**: Rotação, flip, brilho, contraste, zoom

## ✅ Verificação de Ambiente

Antes de começar, verifique se tudo está configurado corretamente:

```bash
python utils/check_setup.py
```

Este script verifica:
- Versão do Python (3.8+)
- Dependências instaladas
- Estrutura do dataset
- Diretórios necessários

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
- Execute: `python utils/check_setup.py` para diagnóstico completo
- Se o dataset não estiver organizado, use: `python data/organize_isic.py --help`

### Erro: "KeyError: 'training'"

Se você encontrar este erro ao executar `train_resnet.py` ou `train_efficientnet.py`, certifique-se de estar usando a versão mais recente dos scripts. Os scripts foram atualizados para incluir a configuração `training` necessária.

### Checkpoints não encontrados

- Execute primeiro o treinamento (`python experiments/main.py` ou scripts individuais)
- Os checkpoints são salvos automaticamente em `./checkpoints/`

## 📝 Exemplo Completo de Execução

### Método Rápido (Recomendado)

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Executar tudo em sequência
python experiments/main.py

# 3. Gerar relatório completo
python experiments/generate_report.py

# 4. Visualizar no TensorBoard
tensorboard --logdir ./runs
```

### Método Manual (Passo a Passo)

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Treinar ResNet-50
python experiments/train_resnet.py

# 3. Treinar EfficientNet-B0
python experiments/train_efficientnet.py

# 4. Comparar modelos
python experiments/compare.py

# 5. Gerar relatório
python experiments/generate_report.py

# 6. (Opcional) Análise de explicabilidade
python experiments/analyze_explainability.py --num_samples 50

# 7. Visualizar no TensorBoard
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
