# Relatório de cobertura de PRs no roadmap (CONE_DETECTION_ROADMAP_AGENT.txt)

Data da análise: 2026-01-17

## Tabela por PR

| PR# | Título | URL | Estado | Cited | Citation quality | Roadmap location | Evidence snippet | Recommended action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Modularize monolithic cone detection script into maintainable package structure (autor: Copilot; criado 2026-01-04; merged 2026-01-04) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/1 | merged | Yes | Cited & detailed | Resumo executivo + Inventário detalhado (modularização e módulos) | L16-19: "A arquitetura é modular, permitindo fácil manutenção e extensão." | Manter; opcional: referenciar explicitamente a PR #1 no histórico do roadmap. |
| 2 | Implement hot-reload configuration and fix premature track deletion (autor: Copilot; criado 2026-01-04; merged 2026-01-04) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/2 | merged | Yes | Mentioned | Módulo app.py (hot-reload) + parâmetros do tracker | L30-31: "Sistema de configuração hot-reload"; L74-81: "Recarrega configuração do arquivo YAML e reinicializa componentes" | Atualizar seção de parâmetros para refletir os novos defaults (lost_timeout, association_max_distance, etc.) e mencionar explicitamente a correção do bug de deleção prematura. |
| 3 | Add Streamlit multi-page dashboard for remote cone tracker configuration via SFTP (autor: Copilot; criado 2026-01-04; merged 2026-01-04) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/3 | merged | No | Not cited | Recomenda-se adicionar seção do dashboard (pasta streamlit_app; não encontrada no tree atual) | — | Incluir uma subseção “Dashboard Streamlit (SFTP)” descrevendo páginas, fluxo de configuração remota e dependências (paramiko/streamlit) e validar onde o código está localizado. |
| 4 | Fix critical SFTP errors and deprecation warnings in Streamlit dashboard (autor: Copilot; criado 2026-01-05; merged 2026-01-05) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/4 | merged | No | Not cited | Recomenda-se na seção do dashboard (streamlit_app ausente no tree atual) | — | Adicionar nota de manutenção do SFTP (stat.S_ISDIR, erros CORS/XSRF, troubleshooting) e verificar se o dashboard foi removido/movido. |
| 5 | Enable dark mode as default theme (autor: Copilot; criado 2026-01-05; merged 2026-01-05) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/5 | merged | No | Not cited | Recomenda-se na seção do dashboard (streamlit_app ausente no tree atual) | — | Adicionar nota de tema padrão no config.toml do Streamlit (dark mode) e impacto visual, após confirmar a localização do app. |
| 6 | Fix ValueError in Streamlit Presets page: add safe numeric formatting utilities (autor: Copilot; criado 2026-01-05; merged 2026-01-05) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/6 | merged | Yes | Mentioned | Testes e validação (lista de testes) | L2628-2632: inclui "test_streamlit_formatting.py" | Documentar utilitários de formatação e o bug corrigido (verificar localização do código Streamlit, ausente no tree atual). |
| 7 | Add video file input support with automatic camera fallback (autor: Copilot; criado 2026-01-08; merged 2026-01-08) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/7 | merged | Yes | Cited & detailed | Módulo app.py (fluxo de captura) + parâmetros camera.video_path | L107-111: verificação de video_path e fallback para câmera | Nenhuma ação crítica; opcional: adicionar referência explícita à PR #7 no histórico. |
| 8 | Add headless video processing with output_video_path configuration (autor: Copilot; criado 2026-01-08; merged 2026-01-08) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/8 | merged | Yes | Cited & detailed | Resumo executivo + Módulo app.py (saída de vídeo) | L31-32: "Saída de vídeo processado para ambientes headless"; L115-118: setup output_video_path | Manter; opcional: reforçar troubleshooting de headless em seção de implantação. |
| 9 | Fix visualizer to always draw annotations and add debug logging for rejections/suspects (autor: Copilot; criado 2026-01-08; merged 2026-01-08) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/9 | merged | Yes | Cited & detailed | Módulo app.py (logs de rejeições/suspects) | L140-150: logs de rejeições/suspects no loop | Manter; opcional: referenciar PR #9 em “Debug e visualização”. |
| 11 | Add frames_seen property to Track class to fix AttributeError in suspect logging (autor: Copilot; criado 2026-01-08; merged 2026-01-08) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/11 | merged | Yes | Cited & detailed | Módulo tracker.py (classe Track) | L739-741: propriedade frames_seen definida | Nenhuma ação necessária. |
| 12 | Add terminal debug output for heading/steering with MP4 video config (autor: Copilot; criado 2026-01-11; merged 2026-01-11) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/12 | merged | Yes | Cited & detailed | Módulo app.py (_debug_print_heading) | L84-100: descrição do HEADING_DBG e cálculo | Nenhuma ação necessária. |
| 13 | Add agent-generated Cone Detection Roadmap (autor: Copilot; criado 2026-01-15; fechado 2026-01-17 sem merge) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/13 | closed | No | Not cited | Recomenda-se seção “Histórico do roadmap” | — | Registrar a tentativa inicial (PR #13, incompleta) para rastreabilidade do documento. |
| 14 | Add comprehensive cone detection roadmap documentation (autor: Copilot; criado 2026-01-15; merged 2026-01-15) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/14 | merged | Yes | Ambiguous | Documento é o próprio roadmap (sem referência explícita à PR) | L1-6: cabeçalho do roadmap (data/autor) | Adicionar “Histórico de versões” citando explicitamente a PR #14 como criação oficial do roadmap. |
| 15 | Add optional debug overlay showing heading direction and angle for confirmed tracks (autor: Copilot; criado 2026-01-15; merged 2026-01-15) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/15 | merged | Yes | Cited & detailed | Módulo visualizer.py (overlay de heading) + parâmetros debug.show_heading_overlay | L1527-1542: overlay com LEFT/RIGHT/CENTER e ângulo | Nenhuma ação necessária. |
| 16 | Document heading overlay visualization in roadmap (autor: Copilot; criado 2026-01-15; merged 2026-01-17) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/16 | merged | Yes | Cited & detailed | Módulo visualizer.py e parâmetros debug.* | L248-250: show_heading_overlay e deadband; L1527-1542: overlay detalhado | Nenhuma ação necessária. |
| 17 | [WIP] Update roadmap document for new debug visualization capability (autor: Copilot; criado 2026-01-15; open) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/17 | open | Yes | Ambiguous | Conteúdo já está no roadmap, mas PR permanece aberta | L1527-1542: overlay de heading (já documentado) | Encerrar/atualizar a PR #17 ou adicionar nota de que o conteúdo já foi incorporado via PR #16. |
| 18 | Enforce camera/video exclusivity and add source-aware run logging (autor: Copilot; criado 2026-01-17; merged 2026-01-17) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/18 | merged | No | Not cited | Recomenda-se seção app.py (validação de fonte) + debug (run log JSONL) | — | Documentar “fonte do input (camera|video)” e logging JSONL/overlay de source (alta prioridade). |
| 19 | [WIP] Analyze PRs in CONE_DETECTION_ROADMAP_AGENT.txt (autor: Copilot; criado 2026-01-17; open) | https://github.com/narutojgdr-sudo/Trekking-DragonBotZv2/pull/19 | open | No | Not cited | Recomenda-se seção “Processo de documentação/relatórios” | — | Adicionar nota de governança sobre auditorias de PRs no roadmap. |

## Resumo geral

- Total de PRs: **18** (IDs 1-9 e 11-19; não há PR #10)
- PRs corretamente documentadas (Cited & detailed): **8**
- PRs parcialmente documentadas (Mentioned / Cited & summarized): **2**
- PRs com citação ambígua: **2**
- PRs não documentadas: **6**

## Debug CSV export (pré-protótipo)

- **Objetivo:** gerar CSVs normalizados por execução para análise antes do protótipo.
- **Local padrão:** `debug.csv_export.csv_dir` (default: `logs/csv`).
- **Formato CSV (header):** `frame_idx,ts_wallclock_ms,ts_source_ms,source,detected,target_id,cx,cy,err_px,err_norm,err_deg,bbox_h,est_dist_m,avg_score,fps`
- **Exemplo de linha:** `123,1674052345678,1674052345000,video,true,3,480.0,270.0,-120.0,-0.250,-10.12,80,2.53,0.78,29.8`
- **Nota:** CSV **NÃO** é protocolo final para ESP32 — é para análise pré‑protótipo.
- **Habilitar:** `debug.csv_export.enabled=true` no `cone_config.yaml`.

## Test scripts: ESP32 integration (tests/exec, tests/esp32)

- **Sender (host):** `tests/exec/send_csv.py` envia linhas CSV via serial e imprime ACKs.
- **Sketch (ESP32):** `tests/esp32/esp32_dev_wroom32/main.ino` lê CSV e responde `ACK`.
- **Teste manual rápido:**
  1. Gere um CSV (`debug.csv_export.enabled=true`).
  2. Flash no ESP32 e conecte via `/dev/ttyUSB0`.
  3. Rode: `python3 tests/exec/send_csv.py --device /dev/ttyUSB0 --csv-file logs/csv/run_video_YYYYMMDD.csv`.

## Recomendações prioritárias (com texto sugerido)

- **[Alta] Atualizar parâmetros críticos do tracker (PR #2)**
  - *Título sugerido:* "Defaults atuais do tracker (pós-hot-reload)"
  - *Texto sugerido:* "Após a correção de deleção prematura (PR #2), alinhar os defaults do tracker ao cone_config.yaml.example para reduzir reinicializações de rastreamento e acelerar a confirmação. Atualize o cone_config.yaml e valide em vídeo real." 

- **[Alta] Documentar source-aware logging e run logs (PR #18)**
  - *Título sugerido:* "Fonte de entrada (camera vs vídeo) e exportação JSONL"
  - *Texto sugerido:* "O app valida exclusividade entre camera.index e camera.video_path e pode exportar JSONL por frame com timestamps e origem. Recomenda-se registrar as chaves debug.export_run_log, debug.run_log_dir e a anotação visual de origem." 

- **[Média] Dashboard Streamlit (PRs #3, #4, #5, #6)**
  - *Título sugerido:* "Dashboard Streamlit para tuning remoto"
  - *Texto sugerido:* "Existe um dashboard multi-página (PRs #3-#6). A pasta streamlit_app não está presente na árvore de arquivos atual. Confirmar se foi removida ou movida. Documentar dependências (streamlit/paramiko), temas (modo escuro) e correções de SFTP/formatação quando o caminho for confirmado." 

- **[Baixa] Histórico de versões do roadmap (PRs #13, #14, #17, #19)**
  - *Título sugerido:* "Histórico do roadmap e auditorias"
  - *Texto sugerido:* "Registrar PR #14 como criação do roadmap, PR #13 como tentativa incompleta e PRs WIP (#17, #19) como auditorias em andamento." 

---
