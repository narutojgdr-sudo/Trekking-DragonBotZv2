# Batch Image Detection Guide

Processe múltiplas imagens de cones em lote com análise detalhada de detecções, rejeições e suspects.

## Configuração Rápida

### 1. Preparar Pasta de Dataset

```bash
mkdir -p DATASET
# Copiar imagens para DATASET/
cp /caminho/para/imagens/*.png DATASET/
cp /caminho/para/imagens/*.jpg DATASET/
```

**Formatos suportados**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`

### 2. Executar Batch Detection

```bash
python3 batch_detect_images.py
```

Ou com parâmetros customizados:

```bash
python3 batch_detect_images.py --dataset DATASET --output BATCH_OUTPUT --config cone_config.yaml
```

### 3. Resultados

A ferramenta gera:
- **Imagens anotadas**: `BATCH_OUTPUT/out_<nome_original>.png` com bounding boxes e informações
- **Relatório JSON**: `BATCH_OUTPUT/detection_report_<timestamp>.json` - dados estruturados completos
- **Relatório TXT**: `BATCH_OUTPUT/detection_report_<timestamp>.txt` - resumo legível

## Saída Esperada

### Imagens de Saída

Cada imagem processada gera `out_<nome>.png` com:
- ✅ **Cones confirmados** em caixa **verde** com ID e score
- ⚠️ **Suspects** em caixa **amarela** com razão
- ❌ **Áreas rejeitadas** exibindo razão da rejeição
- Informações no topo: contagem total de confirmados, suspects, rejeitados

### Exemplo de Filename
```
Input:  home.png
Output: out_home.png

Input:  scene_01.jpg
Output: out_scene_01.png
```

## Informações Detalhadas nos Logs

### Detecções Confirmadas

Cada cone confirmado inclui:
```
ID 1: score=0.85, pos=(480,270), size=(60x45)
      error=+5.23°, distance=0.45m
```

- **ID**: track_id único
- **score**: avg_score (confiança média)
- **pos**: (cx, cy) - centro do bounding box
- **size**: (width, height) em pixels
- **error**: erro de steering em graus (positivo = direita, negativo = esquerda)
- **distance**: distância estimada (se `cone_height_m` configurado)

### Suspects (Rastreamentos Duvidosos)

```
ID 2: score=0.42, frames_seen=3
      Reason: Below confirmation threshold or lost updates
```

- Cones que não atingiram limiar de confirmação
- Ou tracks que perderam detecções recentes

### Rejection Reasons (Áreas Rejeitadas)

Exemplos de razões:
```
3x: Aspect ratio too wide (width/height > 4.0)
2x: Aspect ratio too narrow (height/width > 3.0)
1x: Bounding box too small (area < 400 px²)
1x: HSV color out of range
```

Ajuste `cone_config.yaml` para alterar limiares se muitos falsos positivos/negativos.

## Configuração Recomendada em cone_config.yaml

Para melhor performance em batch:

```yaml
camera:
  process_width: 960    # Qualidade/velocidade (aumentar para imagens HD)
  process_height: 540
  hfov_deg: 70.0       # Ajustar para sua câmera

debug:
  print_heading: true           # Log com detalhe de steering
  cone_height_m: 0.28           # Para estimar distância
  draw_suspects: true           # Desenhar suspects nas outputs
  show_rejection_reason: true   # Mostrar razão de rejeição nos outputs
  csv_export:
    enabled: false      # (desativar para batch)

geometry:
  confirm_avg_score: 0.50  # Limiar de confiança para confirmação
  min_frame_score: 0.30    # Limiar mínimo por frame

tracking:
  min_frames_for_confirm: 4   # Frames necessários para confirmar
  association_max_distance: 250
```

## Exemplos de Uso

### Processar DATASET local
```bash
python3 batch_detect_images.py
```

### Processar pasta customizada
```bash
python3 batch_detect_images.py --dataset ./meus_cones --output ./resultados
```

### Com configuração customizada
```bash
cp cone_config.yaml cone_config_batch.yaml
# Editar cone_config_batch.yaml conforme necessário
python3 batch_detect_images.py --config cone_config_batch.yaml
```

## Interpretando Resultados

### Bom resultado
```
Image: scene.png
Output: out_scene.png
Confirmed: 3, Suspects: 0, Rejected: 2
```
✅ Detecção clara, poucos rejects.

### Resultado problemático
```
Image: dark_scene.png
Output: out_dark_scene.png
Confirmed: 0, Suspects: 5, Rejected: 20
Rejection Reasons:
  15x: HSV color out of range
  5x: Aspect ratio too narrow
```
⚠️ Ajustar HSV range ou brightness em cone_config.yaml.

## Estrutura de Saída

```
BATCH_OUTPUT/
├── out_image1.png
├── out_image2.png
├── out_image3.png
├── detection_report_20260125_143022.json
└── detection_report_20260125_143022.txt
```

## Dicas

1. **Testar com poucas imagens primeiro**:
   ```bash
   mkdir test_dataset && cp DATASET/image1.png test_dataset/
   python3 batch_detect_images.py --dataset test_dataset --output test_output
   ```

2. **Comparar com configurações diferentes**:
   - Crie múltiplos `cone_config_*.yaml` com ajustes HSV/geometry
   - Execute batch para cada e compare resultados

3. **Analisar rejeições**:
   - Abra `.txt` report e procure padrões em rejection reasons
   - Ajuste `hfov_deg`, aspect ratio limits, ou color ranges conforme necessário

4. **Usar relatório JSON para análise**:
   ```python
   import json
   with open('detection_report_*.json') as f:
       data = json.load(f)
   print(f"Total detected: {data['summary']['total_confirmed']}")
   ```

## Troubleshooting

### "No images found in DATASET"
- Verificar pasta DATASET existe
- Extensão de arquivo é suportada (.png, .jpg, etc)?

### Muitas rejeições
1. Imagens muito escuras/claras → ajustar gain/brightness
2. Cor do cone fora do range HSV → editar `cone_config.yaml`
3. Aspect ratio muito restritivo → aumentar `max_aspect_ratio`

### Muitos suspects, poucos confirmados
- Aumentar `confirm_avg_score` em geometry se muito permissivo
- Diminuir se muito restritivo
- Ou ajustar `min_frames_for_confirm` (padrão: 4 frames)

## Para Mais Detalhes

Veja:
- `cone_config.yaml` - todas as opções documentadas
- `ARCHITECTURE.md` - design da pipeline
- `cone_tracker/detector.py` - lógica de detecção HSV
- `cone_tracker/tracker.py` - lógica de rastreamento multi-objeto
