# 🖥️ Interface Gráfica - Sistema de Detecção de Melanoma

## Visão Geral

A interface gráfica (GUI) do sistema oferece uma forma intuitiva e completa para treinar modelos, fazer predições e analisar resultados, tudo através de uma interface web interativa construída com Gradio.

## 🚀 Iniciando a Interface

### Pré-requisitos

Certifique-se de ter instalado todas as dependências:

```bash
pip install -r requirements.txt
```

### Executar a Aplicação

```bash
python app.py
```

A interface será iniciada automaticamente e estará disponível em:
- **Local**: http://localhost:7860
- **Rede**: http://0.0.0.0:7860 (acessível de outros dispositivos na mesma rede)

Para encerrar o servidor, pressione `Ctrl+C` no terminal.

## 📋 Funcionalidades da Interface

### 🏠 Página Inicial

A página inicial apresenta uma visão geral do sistema, suas funcionalidades e instruções básicas de uso.

### 🏋️ Treinamento

**Treine modelos com visualização em tempo real do progresso**

#### Funcionalidades:
- Seleção do modelo (ResNet-50 ou EfficientNet-B0)
- Configuração de hiperparâmetros:
  - Número de épocas
  - Batch size
  - Learning rate
  - Early stopping patience
- Logs de treinamento em tempo real
- Atualização automática do progresso
- Salvamento automático de checkpoints

#### Como usar:
1. Selecione o modelo desejado
2. Configure o caminho dos dados (padrão: `./data/isic2020`)
3. Ajuste os hiperparâmetros conforme necessário
4. Clique em "▶️ Iniciar Treinamento"
5. Acompanhe o progresso na seção de logs
6. Os checkpoints serão salvos automaticamente em `./checkpoints/{modelo}/`

#### Saída:
- Checkpoint do melhor modelo: `./checkpoints/{modelo}/best_model.pth`
- Logs TensorBoard: `./runs/{modelo}/`
- Histórico: `./results/training_history/{modelo}_{timestamp}.json`

### 🔍 Predição Individual

**Analise imagens individuais com visualização de explicabilidade**

#### Funcionalidades:
- Upload de imagem única
- Predição em tempo real
- Visualização de probabilidades (gráfico de barras)
- Grad-CAM para explicabilidade visual
- Comparação entre modelos (ResNet-50 vs EfficientNet-B0)

#### Como usar:
1. **Carregar modelos**:
   - Clique em "Carregar ResNet-50" (checkpoint padrão: `./checkpoints/resnet50/best_model.pth`)
   - Clique em "Carregar EfficientNet-B0" (checkpoint padrão: `./checkpoints/efficientnet_b0/best_model.pth`)
   - Aguarde a confirmação de carregamento

2. **Fazer predição**:
   - Carregue uma imagem de lesão cutânea
   - Selecione o modelo (ou "Comparar ambos")
   - Marque "Mostrar Grad-CAM" para visualização de explicabilidade
   - Clique em "🔮 Fazer Predição"

3. **Interpretar resultados**:
   - **Predição**: Classe predita (Benigno/Maligno)
   - **Probabilidades**: Confiança do modelo em cada classe
   - **Grad-CAM**: Mapa de calor mostrando regiões importantes para a decisão

#### Interpretando Grad-CAM:
- **Regiões vermelhas**: Áreas com alta ativação (importante para a decisão)
- **Regiões azuis/verdes**: Áreas com baixa ativação
- Ajuda a entender se o modelo está focando em características relevantes

### 📁 Predição em Lote

**Processe múltiplas imagens simultaneamente**

#### Funcionalidades:
- Upload de múltiplas imagens
- Processamento em lote
- Exportação de resultados em CSV
- Visualização em tabela

#### Como usar:
1. Selecione o modelo
2. Carregue múltiplas imagens (arquivos .jpg, .png, etc.)
3. Clique em "🚀 Processar Lote"
4. Visualize os resultados na tabela
5. (Opcional) Baixe os resultados em CSV

#### Formato da tabela:
| Imagem | Predição | Prob. Benigno | Prob. Maligno |
|--------|----------|---------------|---------------|
| img1.jpg | Benigno | 85.32% | 14.68% |
| img2.jpg | Maligno | 23.41% | 76.59% |

### ⚖️ Comparação

**Compare o desempenho dos dois modelos**

#### Funcionalidades:
- Comparação lado a lado de métricas
- Gráficos comparativos
- Identificação do melhor modelo por métrica
- Resumo dos melhores modelos treinados

#### Como usar:
1. Certifique-se de que ambos os modelos foram treinados
2. Clique em "🔄 Executar Comparação"
3. Aguarde o processamento
4. Visualize os gráficos de comparação
5. Analise a tabela de melhores modelos

#### Métricas comparadas:
- **Acurácia**: Proporção de acertos
- **Sensibilidade**: Capacidade de detectar malignos
- **Especificidade**: Capacidade de detectar benignos
- **AUC-ROC**: Discriminação geral
- **F1-Score**: Equilíbrio precisão/recall

### 📊 Histórico

**Visualize e analise todos os treinamentos realizados**

#### Funcionalidades:
- Lista de todos os treinamentos
- Curvas de aprendizado
- Evolução de métricas por época
- Comparação de diferentes runs

#### Como usar:
1. A tabela mostrará todos os treinamentos salvos
2. Clique em "🔄 Atualizar Lista" para refresh
3. Selecione um treinamento no dropdown
4. Clique em "📈 Plotar Curvas"
5. Visualize:
   - Curvas de loss (treino e validação)
   - Evolução do AUC-ROC
   - Schedule do learning rate
   - Métricas finais no conjunto de teste

#### Curvas disponíveis:
- **Loss**: Train vs Validation
- **AUC-ROC**: Evolução na validação
- **Learning Rate**: Schedule ao longo do treino
- **Métricas de Teste**: Barras comparativas

### ℹ️ Sobre

Informações detalhadas sobre:
- Arquitetura do projeto
- Tecnologias utilizadas
- Pipeline de treinamento
- Configurações padrão
- Referências bibliográficas

## 🎨 Estrutura da GUI

```
gui/
├── __init__.py
├── training_interface.py        # Interface de treinamento
├── prediction_interface.py      # Interface de predição
├── comparison_interface.py      # Interface de comparação e histórico
└── (gerenciado por app.py)
```

## 💡 Dicas de Uso

### Para Melhor Experiência

1. **Treinamento**:
   - Comece com valores padrão de hiperparâmetros
   - Use early stopping para evitar overfitting
   - Monitore os logs em tempo real
   - Aguarde confirmação de salvamento do checkpoint

2. **Predição**:
   - Use imagens de boa qualidade
   - Sempre carregue os modelos antes de fazer predições
   - Use Grad-CAM para validar se o modelo está focando em regiões corretas
   - Compare ambos os modelos para maior confiança

3. **Análise**:
   - Explore o histórico para identificar melhores configurações
   - Compare métricas clínicas (sensibilidade é crucial para melanoma!)
   - Use as curvas de aprendizado para diagnosticar overfitting

### Compartilhar na Rede

Para permitir acesso de outros dispositivos na mesma rede:

1. Identifique seu IP local:
   ```bash
   # Windows
   ipconfig

   # Linux/Mac
   ifconfig
   ```

2. Outros dispositivos podem acessar em:
   ```
   http://<SEU_IP>:7860
   ```

### Criar Link Público (Temporário)

Modifique [app.py](app.py:303) linha 303:

```python
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Mude para True
    show_error=True,
    show_api=False
)
```

Isso gerará um link público temporário (válido por 72h).

## 🐛 Solução de Problemas

### Erro: "Modelo não carregado"
**Causa**: Checkpoint não encontrado
**Solução**: Verifique o caminho do checkpoint ou treine o modelo primeiro

### Erro: "CUDA out of memory"
**Causa**: GPU sem memória suficiente
**Solução**: Reduza o batch size ou use CPU (será detectado automaticamente)

### Logs não atualizam
**Causa**: Auto-refresh pode ter falhado
**Solução**: Clique em "🔄 Atualizar Logs" manualmente

### Grad-CAM não aparece
**Causa**: Modelo não carregado corretamente ou erro no processamento
**Solução**: Recarregue o modelo e tente novamente

### Interface travada
**Causa**: Processo longo em execução
**Solução**: Aguarde conclusão ou reinicie o servidor

## 📊 Monitoramento Avançado

Para análise mais detalhada durante treinamento, use TensorBoard em paralelo:

```bash
tensorboard --logdir=./runs
```

Acesse em: http://localhost:6006

## 🔒 Considerações de Segurança

- **Não use em produção médica**: Este sistema é para fins educacionais
- **Validação médica necessária**: Resultados devem ser validados por profissionais
- **Dados sensíveis**: Certifique-se de seguir regulamentações de privacidade (LGPD, HIPAA)
- **Acesso à rede**: Limite o acesso ao servidor se processar dados reais

## 📝 Notas de Desenvolvimento

### Customização

Você pode customizar a interface modificando:
- **Tema**: Altere `theme=gr.themes.Soft()` em [app.py](app.py:220)
- **Porta**: Modifique `server_port=7860` em [app.py](app.py:303)
- **CSS**: Adicione estilos customizados na seção `css` em [app.py](app.py:221-226)

### Adicionar Novas Funcionalidades

1. Crie novo arquivo em `gui/`
2. Implemente a interface usando Gradio Blocks
3. Importe e adicione uma nova tab em [app.py](app.py:279-295)

## 🤝 Contribuindo

Sugestões de melhorias para a GUI:
- [ ] Adicionar suporte a vídeo/webcam para captura ao vivo
- [ ] Implementar A/B testing entre modelos
- [ ] Adicionar exportação de relatórios em PDF
- [ ] Integrar análise estatística avançada
- [ ] Suporte multilíngue

## 📚 Recursos Adicionais

- [Documentação Gradio](https://www.gradio.app/docs/)
- [Tutorial PyTorch](https://pytorch.org/tutorials/)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [ISIC Challenge](https://challenge.isic-archive.com/)

---

**Desenvolvido com Gradio 4.0+ | Atualizado: 2026**
