# CAD AI — WORK 01: Vertical Slice determinista V0.1

Repositorio mínimo que demuestra una cadena CAD real y determinista. M3-A añade la frontera para un planner LLM no confiable, pero todavía no instala ni ejecuta ningún modelo real:

`DesignSpec manual → CADPlan → CadQuery/OpenCascade → CADResult + FeatureRegistry → GeometryInspector → Validator → ValidationReport → STEP/STL`

El Hito 2 añade un primer ciclo de revisión determinista: `R01 → FAIL → RepairPatch → R02 → PASS`.

M2 añade una frontera explícita de reparación estructurada:

`ValidationReport → DeterministicRepairPlanner → RepairPlan → RepairPlanValidator → RepairExecutor → CADPlan R02`

M3-A extiende únicamente el lado de planificación:

`RepairPlanningContext → RepairPlanner → RepairPlannerResponse → RepairPlanValidator → RepairExecutor`

## Qué implementa

- Contratos JSON tipados con Pydantic y `contract_version = "0.1"`.
- `DesignSpec` de requisitos y `CADPlan` de operaciones resueltas y controladas.
- Operaciones `box`, `cylinder` y `hole` sobre CadQuery/OpenCascade.
- Features semánticos estables y locators internos sin índices topológicos públicos.
- `CADResult` con fronteras `SUCCESS/ERROR` y errores CAD estructurados.
- Inspección de validez, sólidos, extents, volumen, distancia entre datums y diámetro cilíndrico real.
- Constraints `HARD/SOFT`, comparaciones `EQ/GTE/LTE/BETWEEN` y reportes `PASS/FAIL/INCOMPLETE`.
- Exportación de la geometría final a STEP y STL.
- Cinco casos deterministas, contract tests, tests geométricos e integración end-to-end.
- Reparación dirigida de un constraint dimensional `HARD + EQ` vinculado explícitamente a un parámetro CAD.
- Conservación en memoria de R01 y R02, sus resultados, reports y artifacts independientes.
- Comprobaciones objetivas iniciales de `CHANGE_LOCALITY` y `CONSTRAINT_PRESERVATION`.
- `RepairPlan V0.1` serializable, limitado a acciones `SET_PARAMETER`.
- Validación semántica independiente con errores estructurados y protección contra planes obsoletos.
- Executor determinista que no interpreta el ValidationReport ni ejecuta CadQuery.
- `RepairPlanningContext` mínimo: failures, constraints protegidos, operaciones/parámetros permitidos y tipos de acción autorizados.
- Interfaz común de planner con baseline determinista y `LLMRepairPlanner` independiente del proveedor.
- `RepairPlannerResponse` discriminado en `PLAN`, `UNSUPPORTED` y `ERROR`.
- Backend fake que atraviesa parsing real y adapter HTTP genérico compatible con servidores estilo `llama-server`.
- Retries acotados, errores estructurados y metadata `PlannerRun`, sin almacenar chain-of-thought.
- Benchmark reproducible RP01–RP07 con geometría y revalidación reales para los casos reparables.

## Fuera de alcance

No incluye modelos/pesos LLM, selección de runtime/modelo, agentes, código CAD generado, tool calling CAD, reparación generativa, UI, base de datos, assemblies complejos, reconocimiento universal de features, FEM ni slicer.

## Requisitos e instalación

Probado con Python 3.12. Se admite Python 3.11–3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[test]"
```

CadQuery instala OpenCascade mediante su stack OCP. En plataformas donde `pip` no resuelva sus wheels, use un entorno Conda compatible con CadQuery y después instale el proyecto editable.

## Tests

```bash
pytest
```

Los tests crean geometría real; no sustituyen las comprobaciones importantes por mocks.

## M4 V0.1 — generación inicial

M4 añade un único vertical slice sin modificar M0–M3:

```text
User Prompt → LLM M4Intent → DesignSpec → CADPlan R01 → CAD Engine → Validator → STEP/STL
```

El LLM produce únicamente `M4Intent` (`width`, `depth`, `height`,
`hole_diameter`, `centered`) mediante structured output. Un primer compilador
determinista crea la `DesignSpec` canónica, incluidos constraints, IDs,
prioridades, measurements y referencias semánticas. El compilador existente
obtiene después `box(x,y,z)` de `EXTENT_X/Y/Z` y crea un único agujero pasante
centrado usando `DIAMETER`. `CADPlan R01`, la ejecución CAD y la validación
geométrica permanecen bajo control determinista. No existe fallback ni
integración con el Repair Planner.

Con `llama-server` escuchando en el puerto 8080:

```bash
python -m cad_ai.m4 \
  --base-url http://127.0.0.1:8080 \
  --model-id cad-ai-m3-qwen35-9b-q6 \
  --output artifacts/m4/M4-001 \
  --result-json artifacts/m4/M4-001/result.json \
  --json
```

El prompt predeterminado es el caso `M4-001`: placa de 60 × 40 × 4 mm con un
agujero pasante central de 6 mm. Los artifacts quedan en
`artifacts/m4/M4-001/R01/`.

## M5 V0.1 — generación con una reparación acotada

M5 integra el Repair Loop M2/M3 sin ampliar el subconjunto geométrico de M4:

```text
M4 generation → CADPlan R01 → CAD Engine → Validator
  ├─ PASS → R01 VALIDATED
  └─ FAIL → RepairPlanningContext → LLMRepairPlanner
           → RepairPlanningBoundary → CADPlan R02 → Validator
```

Si R01 pasa, el Repair Planner no se invoca. Si falla, se permite como máximo
una propuesta y una revisión R02. `RepairPlanValidator` conserva la autoridad
de autorización y `RepairExecutor` es el único componente que aplica el plan.
`UNSUPPORTED`, `ERROR` o un plan rechazado terminan estructuradamente, sin
fallback. Se mantiene como limitación conocida que Qwen3.5-9B Q6_K puede
devolver `UNSUPPORTED` para una reparación válida.

El parámetro interno `test_plan_transform` de `run_m5_001()` permite introducir
una desviación reproducible después de compilar el plan correcto. No está
expuesto por CLI y se usa exclusivamente en tests/benchmarks.

Ejecución real normal, sin inyección:

```bash
python -m cad_ai.m5 \
  --base-url http://127.0.0.1:8080 \
  --model-id cad-ai-m3-qwen35-9b-q6 \
  --output artifacts/m5/M5-001 \
  --result-json artifacts/m5/M5-001/result.json \
  --json
```

Caso controlado R01 FAIL → R02:

```bash
pytest -q tests/test_m5.py -k valid_plan
```

## CAD AI V0.2 — Capability Pack 1

V0.2 amplía verticalmente la familia de placas sin cambiar las fronteras de
autoridad de M0–M5:

```text
Prompt → DeterministicPromptGrounder → GroundedPromptEvidence → ExtractionCoverage
  COMPLETE → DeterministicFactAssembler ───────────────────────────────┐
  PARTIAL  → ResidualLLMExtractor → ResidualFacts → FactAssembler ─────┤
                                                                        ↓
             FactGroundingValidator (solo híbrido) → DeterministicPlateIntentGate
             → PlateIntent V0.2 → DesignSpec → CADPlan R01 → CAD Engine → Validator
                                                                └─ FAIL → M3 → R02
```

La frontera es *evidence-first*: la evidencia léxica determinista tiene
precedencia sobre cualquier salida del LLM. Cuando cubre por completo la
petición, el sistema ensambla `ExtractedPlateFacts` canónicos sin invocar un
modelo (`extraction_mode=DETERMINISTIC`). Si solo faltan campos concretos, un
schema reducido permite que el LLM devuelva exclusivamente esos valores
(`extraction_mode=HYBRID_LLM`); no puede modificar hechos grounded ni producir
`unsupported_features`. `FactGroundingValidator` comprueba ese camino híbrido.
El Gate determinista decide
`INTENT`/`UNSUPPORTED`, normaliza un único agujero
`centered` a `(0,0)` y aplica las reglas de dimensiones, posiciones y conflicto
fillet/chamfer. Desde `PlateIntentResponse` el pipeline existente permanece
sin cambios.

Capability Pack 1 acepta una placa rectangular, uno o más agujeros pasantes con
diámetro y coordenadas X/Y explícitas, y opcionalmente un único fillet o
chamfer sobre las cuatro aristas verticales exteriores. El compilador asigna
IDs `H01`, `H02`, … ordenando por `(x, y, diameter)`, por lo que los IDs no
dependen del orden de la lista recibido del LLM. Cada agujero produce una
operación `hole`, un feature `hole_hNN` y un datum `hole_hNN_axis`.

Las nuevas mediciones deterministas son `FEATURE_EXISTS`, `HOLE_COUNT`,
`POSITION_X`, `POSITION_Y`, `FILLET_RADIUS` y `CHAMFER_DISTANCE`. Fillet y
chamfer usan la selección semántica fija `OUTER_VERTICAL_EDGES`; no se exponen
índices OCC. La combinación fillet + chamfer en una misma pieza queda fuera de
Capability Pack 1.

Ejecuciones reales:

```bash
python -m cad_ai.v02 CASE-02 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CASE-04 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CASE-05 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
```

Los artifacts se escriben en `artifacts/v02/<CASE>/R01/` o `R02/`; el
`result.json` distingue siempre `final_revision_id` y `final_artifacts`. La
inyección de fallos sigue siendo exclusivamente programática para tests y
benchmarks, nunca una opción del CLI normal.

## CAD AI V0.2 — Capability Pack 2

Capability Pack 2 conserva exactamente la frontera `Prompt → Facts → Gate`
y amplía el subconjunto determinista a prismas rectangulares de un solo cuerpo
con features sustractivas sobre la cara superior:

- cero o más agujeros pasantes explícitos;
- pockets rectangulares y circulares, ciegos o through;
- cutouts rectangulares through;
- slots rectos horizontales (`0°`) o verticales (`90°`);
- patrones lineales de agujeros sobre X/Y, anclados por `START` o `CENTER`;
- fillet o chamfer sobre las cuatro aristas verticales exteriores.

El Grounder conserva la compatibilidad diagnóstica con el contrato factual
`capability-facts-3.0`, pero el camino activo usa cobertura y ensamblado
deterministas. `COMPLETE` incluye los casos explícitos CP1/CP2; `PARTIAL`
identifica rutas pendientes concretas y construye para ellas un JSON Schema
mínimo; `INSUFFICIENT` detiene el flujo sin inventar hechos. Un grupo como
`5 mm through holes at (...)` se representa una sola vez mediante
`hole_groups`; el diámetro compartido se aplica determinísticamente a sus
posiciones. Un patrón permanece como pattern factual hasta su expansión
determinista. Los términos explícitos fuera del pack (`thread`, `shell`, etc.)
proceden exclusivamente del Grounder. El Gate valida dimensiones positivas, posición,
profundidad menor que la altura para pockets ciegos, footprint dentro de la
base, duplicados exactos y conflicto fillet/chamfer. Un patrón se expande antes
del `DesignSpec` a agujeros explícitos `H01…HNN`, ordenados canónicamente por
`(x,y,diameter)`.

El orden reproducible del CADPlan es:

```text
box → holes → rectangular pockets → circular pockets → slots → fillet/chamfer
```

Los IDs públicos son semánticos: `rect_pocket_rpNN`,
`circle_pocket_cpNN`, `slot_sNN` y sus datums `_center`/`_axis`. El CAD Engine
añade únicamente operaciones `CONTROLLED`; el LLM nunca recibe CadQuery ni
índices `FaceN/EdgeN`.

El Validator comprueba geometría/locators producidos por la ejecución real:
existencia, counts, posición, diámetro, ancho/profundidad, cut depth, through,
longitud/ancho/orientación del slot y los tratamientos de borde. El Repair Loop
puede autorizar `SET_PARAMETER` localizado sobre esas dimensiones cuando hay
una `RepairRule`; no repara inserciones/eliminaciones estructurales.

## CAD AI V0.2 — Capability Pack 3: additive features

Capability Pack 3 conserva la misma frontera evidence-first y añade únicamente
features aditivas controladas sobre `support_face=TOP`. El sistema de
coordenadas no cambia: el cuerpo base ocupa `z=0…base_height` y toda feature
aditiva ocupa `z=base_height…base_height+feature_height`.

El subset soportado es:

- boss rectangular (`width`, `depth`, `height`, `x`, `y`);
- boss cilíndrico (`diameter`, `height`, `x`, `y`);
- standoff semántico, con agujero axial through opcional;
- grupos explícitos de standoffs con dimensiones compartidas;
- patrones lineales de bosses cilíndricos o standoffs sobre X/Y, con anclaje
  `START` o `CENTER`;
- combinaciones con holes, pockets, slots y fillet/chamfer de CP1/CP2.

El Grounder reconoce solamente sintaxis explícita. Los patrones y grupos
permanecen compactos en `GroundedPromptEvidence`/`ExtractedPlateFacts`; el Gate
los expande determinísticamente antes de crear `DesignSpec`. Para cuatro
instancias con spacing 20 y centro `(0,0)` sobre X, las posiciones son siempre
`(-30,0), (-10,0), (10,0), (30,0)`. El LLM nunca calcula estas coordenadas ni
puede sobrescribir dimensiones grounded. Un prompt completo evita el LLM; un
prompt parcial entrega al extractor residual solo los campos pendientes.

Los IDs públicos se asignan tras orden canónico, no por orden textual:
`rect_boss_rbNN`, `cyl_boss_cbNN`, `standoff_stNN`, sus datums `_center` o
`_axis`, y para standoffs perforados `standoff_stNN_hole` más
`standoff_stNN_hole_axis`. El plan usa operaciones `CONTROLLED` `rect_boss`,
`cyl_boss` y `standoff`; todas fusionan contra `main_body` y deben producir
exactamente un sólido.

El Gate exige dimensiones positivas, footprint dentro de la base, posición
explícita o `centered`, `hole_diameter < outer_diameter`, expansión de patrón
válida, ausencia de centros duplicados y ausencia de solapes evidentes entre
footprints aditivos. Bosses/standoffs sobre caras laterales, patterns
circulares, múltiples cuerpos, threads, shell, revolve, loft y sweep siguen
fuera de alcance.

El Validator añade counts separados y mediciones feature-aware:
`RECTANGULAR_BOSS_COUNT`, `CYLINDRICAL_BOSS_COUNT`, `STANDOFF_COUNT`,
`FEATURE_HEIGHT`, `BASE_HEIGHT`, `OUTER_DIAMETER` e `INNER_DIAMETER`, además de
existencia/posición/ancho/profundidad/diámetro ya disponibles. Las mediciones
verifican que la feature registrada corresponde a sólido o vacío real. Las
`RepairRule` permiten solo `SET_PARAMETER` localizado sobre dimensiones y
posición; no crean, borran ni convierten features.

Ejecuciones reales representativas (los casos `COMPLETE` no invocan Qwen):

```bash
python -m cad_ai.v02 CP3-02 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CP3-03 --prompt "Create an 80 x 60 x 5 mm plate with a 16 mm diameter cylindrical boss 10 mm high whose centre is X 20 mm and Y -10 mm." --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CP3-06 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CP3-07 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CP3-08 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
```

Los artifacts quedan en `artifacts/v02/CP3-NN/R01/`; una reparación controlada
aceptada usa `R02/`. CP3 queda listo para validación real, pero los tests
internos por sí solos no equivalen a validación final con lenguaje natural.

## Prototype Usability Gate V0.1

`cad_ai.prototype_gate` evalúa generalización dentro de Capability Pack 2 sin
añadir geometrías. Sus 20 prompts son distintos de los golden cases: 11
`COMPLETE`, 4 `PARTIAL`, 2 `INSUFFICIENT` y 3 `UNSUPPORTED`. Cada ejecución
compara la cadena completa:

```text
Prompt ↔ CanonicalFacts ↔ PlateIntent ↔ DesignSpec ↔ CADPlan ↔ CAD ↔ Validator
```

Por caso se escriben `result.json` (evaluación de producto) y
`pipeline_result.json` (traza completa). El agregado queda en `summary.json`.
Un resultado geométrico `VALIDATED` con facts, intent, spec o plan distintos de
la expectativa se contabiliza explícitamente como `FALSE_VALIDATED`.

Benchmark completo con respuestas residuales controladas:

```bash
python -m cad_ai.prototype_gate \
  --backend controlled \
  --output artifacts/prototype_gate_v01 \
  --json
```

Filtros reproducibles:

```bash
python -m cad_ai.prototype_gate --classification complete --backend controlled
python -m cad_ai.prototype_gate --classification partial --backend controlled
python -m cad_ai.prototype_gate --classification insufficient --backend controlled
python -m cad_ai.prototype_gate --classification unsupported --backend controlled
```

Solo el camino `PARTIAL` contra Qwen real/llama-server:

```bash
python -m cad_ai.prototype_gate \
  --classification partial \
  --backend http-llm \
  --base-url http://127.0.0.1:8080 \
  --model-id cad-ai-m3-qwen35-9b-q6 \
  --output artifacts/prototype_gate_v01_qwen_partial \
  --json
```

Los casos `COMPLETE`, `INSUFFICIENT` y `UNSUPPORTED` usan un backend que falla
si fuese llamado, demostrando `llm_invoked=false`. El camino híbrido conserva
telemetría de modelo, versión del prompt residual, tokens, latencia,
tokens/segundo y validez de schema cuando el backend la proporciona. La
estabilización final reconoce negaciones explícitas y acotadas como
`thread-free`, `without threads`, `non-threaded`, `no fillet` y sus equivalentes
documentados antes de aplicar la detección lexical positiva. El Gate completo
controlado queda en 20/20, sin convertir esta evidencia en una afirmación de
CAD generalista.

Ejecuciones reales con Qwen/llama-server:

```bash
python -m cad_ai.v02 CP2-02 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CP2-04 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CP2-05 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CP2-06 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
python -m cad_ai.v02 CP2-07 --base-url http://127.0.0.1:8080 --model-id cad-ai-m3-qwen35-9b-q6 --json
```

Artifacts: `artifacts/v02/CP2-XX/R01/*.step|*.stl`; si se ejecuta la reparación
controlada CP2-08 desde tests, el resultado final queda en `R02/`.

Limitaciones deliberadas: slots solo a `0°/90°`; no existe solver general de
intersecciones entre pockets/slots (los solapes booleanos modelables se
permiten); no hay circular patterns, caras laterales, threads, shell, sketches
libres, múltiples cuerpos ni reparación estructural.

## Benchmark M3-A

```bash
python -m cad_ai.benchmark.m3 --planner deterministic
python -m cad_ai.benchmark.m3 --planner fake-llm
python -m cad_ai.benchmark.m3 --planner fake-llm --case RP03 --json
```

El default sigue siendo la suite histórica `M3-RP-v0.1` (identificador de
resultado `m3-a.1`). La revisión se selecciona explícitamente y escribe
artifacts en un destino separado:

```bash
python -m cad_ai.benchmark.m3 \
  --suite M3-RP-v0.2 \
  --planner deterministic \
  --output artifacts/benchmark_m3/M3-RP-v0.2/deterministic
```

M3-RP-v0.2 conserva RP01–RP04, corrige la abstención segura de RP05 y sustituye
RP06/RP07 por RP06A/RP06B y RP07A/RP07B. Su contrato normativo está en
`docs/M3-RP-v0.2.md`.

Para comparar modelos locales sin cambiar fixtures, prompt ni configuración de
generación, usa `--model-id` (el alias anterior `--model` sigue disponible):

```powershell
.\.venv\Scripts\python.exe -m cad_ai.benchmark.m3 `
  --planner http-llm `
  --model-id cad-ai-m3-ministral3-14b-q5km `
  --output artifacts/benchmark_m3/ministral3-14b-q5km `
  --json
```

Antes de ejecutar RP01 contra llama-server real, comprueba que su constrained
generation está activa con el canary independiente del benchmark:

```bash
python -m cad_ai.benchmark.m3 --planner http-llm --canary --json
```

El canary devuelve `PASS` únicamente cuando el schema impuesto por el transporte
vence a una instrucción textual deliberadamente incompatible. No ejecuta CAD ni
ninguno de los casos RP01–RP07.

Por defecto, los casos ejecutados producen artifacts bajo `artifacts/benchmark_m3/<case>/R01` y, cuando hay una reparación autorizada, `R02`. `--output RUTA` cambia ese destino. Los planners determinista y fake atraviesan la misma interfaz y el mismo harness; no existe fallback silencioso entre ellos.

## Demo

```bash
python -m cad_ai.demo pass
python -m cad_ai.demo fail
python -m cad_ai.demo semantic
python -m cad_ai.demo soft
python -m cad_ai.demo error
python -m cad_ai.demo repair
```

Puede cambiar el destino con `--output RUTA`. Por defecto los STEP/STL aparecen en `artifacts/<case>/`. El caso `error` termina en un `CADResult.ERROR` y, correctamente, no genera un `ValidationReport` ni artifacts.

La demo `repair` muestra además el `RepairPlan`, su resultado `VALID` y la ejecución autorizada. Genera directorios separados `artifacts/repair/R01/` y `artifacts/repair/R02/`; no sobrescribe la primera revisión.

## Bucle determinista de revisión

El reparador consume el `ValidationReport` de R01, busca todos sus `HARD FAIL` y solo actúa si cada fallo tiene una regla explícita y soportada. La regla actual relaciona `C_WIDTH` con `OP01.params.x`. El valor nuevo procede del `target` numérico del constraint, no de una condición especial sobre el número de revisión.

`RevisionChange` registra `change_id`, causa, constraint, target estructurado, valores anterior/nuevo, operación y features relacionados. `RepairResult` expresa siempre `REPAIRED`, `UNSUPPORTED`, `NO_CHANGE` o `ERROR`.

- `CHANGE_LOCALITY`: al ignorar únicamente `revision_id` y `parent_revision_id`, los campos semánticos modificados deben coincidir exactamente con los targets declarados en el patch.
- `CONSTRAINT_PRESERVATION`: cada constraint que estaba `PASS` en R01 debe continuar `PASS` en R02.

## M2 — Structured Repair Boundary

M2 separa tres responsabilidades:

- `DeterministicRepairPlanner`: lee `DesignSpec`, `CADPlan R01` y `ValidationReport R01`; propone un `RepairPlan`, pero no modifica el CADPlan.
- `RepairPlanValidator`: autoriza o rechaza el plan por identidad, revisión, targets, valores, constraints, features y permisos.
- `RepairExecutor`: aplica exclusivamente acciones `SET_PARAMETER` de un plan `VALID`; produce `CADPlan R02` y los `RevisionChange` realmente ejecutados.

`RepairPlan V0.1` contiene:

```text
contract_version
design_id / spec_version
source_revision_id / target_revision_id
actions[]
  action_id
  type = SET_PARAMETER
  target.operation_id / target.parameter
  old_value / new_value
  constraint_ids[] / related_feature_ids[]
  reason
rationale
```

El schema no permite código, comandos, callbacks, mutaciones del `DesignSpec` ni acciones arbitrarias. La validación semántica devuelve `VALID` o `INVALID` con códigos como `UNKNOWN_OPERATION`, `UNKNOWN_PARAMETER`, `STALE_OLD_VALUE`, `UNKNOWN_CONSTRAINT`, `REVISION_MISMATCH` y `ACTION_NOT_ALLOWED`.

La validación `VALID` queda vinculada mediante digests al `RepairPlan` y al CADPlan fuente concretos. Si cualquiera cambia antes de ejecutar, el Executor rechaza la operación.

El productor de M2 sigue siendo determinista. M3-A añade un productor LLM provider-independent probado con backend fake; todavía no hay modelo real. En el siguiente hito un runtime local podrá entregar el mismo JSON, mientras RepairPlanValidator, RepairExecutor, CAD Engine y Validator geométrico permanecen bajo control determinista.

## M3-A — LLM Repair Planner Boundary

`RepairPlanningContextBuilder` proyecta desde `DesignSpec`, `CADPlan` y `ValidationReport` únicamente información reparable: IDs de diseño/revisión, `failed_constraints`, `protected_constraints`, operaciones y parámetros autorizados, `allowed_action_types`, un error breve del intento anterior y un hint opcional marcado como no autoritativo. No expone modelos OCC, archivos, código CadQuery, historial de conversación ni paths del filesystem.

Todo planner M3 produce la misma unión discriminada `RepairPlannerResponse V0.1`:

- `PLAN`: contiene exactamente el `RepairPlan V0.1` de M2.
- `UNSUPPORTED`: decisión válida y no reintentable con `reason_code` estructurado.
- `ERROR`: fallo de parsing/runtime con códigos como `MODEL_TIMEOUT`, `MODEL_UNAVAILABLE`, `OUTPUT_PARSE_ERROR`, `SCHEMA_VALIDATION_ERROR` y `RETRY_EXHAUSTED`.

La frontera de confianza es estricta:

```text
raw model output (untrusted)
  → JSON parse
  → RepairPlannerResponse schema
  → RepairPlanValidator semantic authorization
  → RepairExecutor deterministic application
```

Que un JSON sea válido contra el schema no lo autoriza. Operaciones/parámetros inexistentes, valores antiguos obsoletos, identidades incoherentes, constraints/features desconocidos, acciones prohibidas o targets duplicados/conflictivos se rechazan antes del Executor. El CADPlan fuente permanece intacto.

`LLMRepairPlanner` depende de una interfaz pequeña `InferenceBackend`. `ScriptedFakeBackend` prueba todo el pipeline de parsing sin saltárselo. `HTTPInferenceBackend` usa configuración neutral (`base_url`, endpoint, model ID opcional, timeout, temperature, max tokens, seed y JSON schema opcional) y no contiene conocimiento de Qwen, Ministral, gpt-oss ni cuantizaciones.

La política inicial permite como máximo dos intentos. Reintenta timeouts, errores transitorios del modelo y outputs no parseables/schema-invalid; el segundo contexto incluye `previous_attempt_error`. No reintenta `UNSUPPORTED` ni `MODEL_UNAVAILABLE`. Una respuesta `PLAN` semánticamente inválida se rechaza en la frontera autorizadora y tampoco se ejecuta ni se repara especulativamente.

El benchmark versionado `m3-a.1` incluye:

- RP01: reparación de width.
- RP02: reparación de height.
- RP03: diámetro de agujero ligado a feature semántico.
- RP04: parámetros distractores y cambio mínimo.
- RP05: reparación estructural no soportada.
- RP06: target ambiguo, correctamente `UNSUPPORTED`.
- RP07: hint adversarial no autoritativo ignorado frente a datos estructurados.

Calcula `SCHEMA_VALID`, `ACTION_CORRECT`, `TARGET_CORRECT`, `OLD_VALUE_CORRECT`, `NEW_VALUE_CORRECT`, `CONSTRAINT_REF_CORRECT`, `INVENTED_ACTION`, `UNNECESSARY_MODIFICATIONS`, `REPAIR_SUCCESS`, `CHANGE_LOCALITY`, `CONSTRAINT_PRESERVATION`, `EXACT_REPAIR_PLAN` y un `PLANNER_CASE_SUCCESS` que permite considerar correcto el `UNSUPPORTED` esperado de RP05/RP06.

## M3-RP-v0.2 — benchmark revisado

La revisión formaliza dos ejes independientes: `allowed_action_types` autoriza
tipos de acción y `repairable_parameters` enumera los únicos targets autorizados
para `SET_PARAMETER`. `constraint_ids` solo expresa relación con un requisito;
no autoriza ni identifica por sí solo un target. `EXTENT_X/Y/Z`, `DIAMETER` y
`RADIUS` pueden desambiguar por semántica directa. Solo existe ambigüedad cuando,
después de filtrar por autorización, relación y semántica estructurada, siguen
quedando al menos dos targets compatibles.

La suite contiene nueve casos: RP01–RP05, RP06A, RP06B, RP07A y RP07B. RP05
acepta las dos abstenciones seguras `NO_ALLOWED_REPAIR` y `REQUIRES_REPLAN`;
RP06A exige `EXTENT_X → x`; RP06B exige `AMBIGUOUS_TARGET`; RP07B debe producir
el mismo target autorizado que RP07A aunque incluya un hint adversarial no
autoritativo.

Para conservar cada ejecución como JSON y comparar casos/métricas entre suites
y modelos:

```bash
python -m cad_ai.benchmark.m3 --suite M3-RP-v0.2 --planner http-llm \
  --model-id MODEL --result-json artifacts/benchmark_m3/MODEL-v0.2.json --json

python -m cad_ai.benchmark.comparison \
  artifacts/benchmark_m3/qwen-v0.1.json \
  artifacts/benchmark_m3/qwen-v0.2.json \
  artifacts/benchmark_m3/ministral-v0.1.json \
  artifacts/benchmark_m3/ministral-v0.2.json --json
```

## M3-B — Real Local LLM Integration

M3-B conecta el mismo `LLMRepairPlanner` a un servidor HTTP real sin introducir rutas especiales después del planner. La configuración de laboratorio predeterminada es:

```text
base_url: http://127.0.0.1:8080
endpoint: /v1/chat/completions
model: cad-ai-m3-qwen35-9b-q6
runtime: llama.cpp b10985
temperature: 0
seed: 12345
max_tokens: 1024
max_attempts: 1
chat_template_kwargs.enable_thinking: false
```

El primer smoke benchmark debe limitarse a RP01:

```bash
python -m cad_ai.benchmark.m3 \
  --planner http-llm \
  --case RP01 \
  --output artifacts/benchmark_m3/http-llm-rp01 \
  --json
```

`max_attempts=1` significa que el benchmark no reintenta, no corrige JSON y no activa fallback. Un fallo HTTP, de parsing, de schema o de autorización queda registrado como resultado real.

Cada resultado JSON conserva `PlannerRun`, `RepairPlannerResponse`, validación semántica, estado del Executor y estado final del Validator. La telemetría opcional del backend incluye estado HTTP/backend, runtime, tokens de prompt/completion/total, tiempo de prompt, tiempo de generación y tokens por segundo. Estos datos permanecen fuera de los contratos CAD.

## LAB-001 — reparación real de diámetro

LAB-001 es el primer experimento reproducible del laboratorio. Construye una
placa de `60 × 40 × 4 mm` con un agujero central. El requisito permanece en
`Ø6.0 ± 0.01 mm`, mientras el CADPlan R01 usa deliberadamente `Ø5.4 mm`.
El Validator debe encontrar un único blocking FAIL (`C_HOLE_DIAMETER`). Qwen
propone un `RepairPlan`; el plan atraviesa sin excepciones la frontera M3
existente y, si es autorizado, el Executor crea R02 cambiando exclusivamente
`OP02.params.diameter`.

Primero comprueba el constrained output del servidor:

```powershell
.\.venv\Scripts\python.exe -m cad_ai.benchmark.m3 `
  --planner http-llm `
  --model-id cad-ai-m3-qwen35-9b-q6 `
  --canary `
  --json
```

Después ejecuta el experimento real:

```powershell
.\.venv\Scripts\python.exe -m cad_ai.lab LAB-001 `
  --base-url http://127.0.0.1:8080 `
  --model-id cad-ai-m3-qwen35-9b-q6 `
  --output artifacts/lab/LAB-001 `
  --result-json artifacts/lab/LAB-001/result.json `
  --json
```

El laboratorio declara `VALIDATED` únicamente cuando R02 produce un
`ValidationReport.PASS`, `parent_revision_id=R01`, change locality exacto,
constraint preservation, los mismos feature IDs semánticos y STEP/STL de ambas
revisiones. Un `UNSUPPORTED`, error/JSON inválido, rechazo semántico o plan
incorrecto produce `FAIL`; indisponibilidad o timeout del servidor produce
`BLOCKED`. No existen retries, corrección automática ni fallback determinista.

Artifacts esperados:

```text
artifacts/lab/LAB-001/
├── R01/LAB-001_R01.step
├── R01/LAB-001_R01.stl
├── R02/LAB-001_R02.step
├── R02/LAB-001_R02.stl
└── result.json
```

Tests específicos:

```bash
python -m pytest -q tests/test_lab001.py
```

## Estructura

```text
src/cad_ai/contracts.py   DesignSpec, constraints, CADPlan y operaciones
src/cad_ai/cad.py         CAD Engine, FeatureRegistry, CADResult, inspector y exportación
src/cad_ai/validation.py  CheckResult, ValidationReport y Validator
src/cad_ai/cases.py       Casos manuales deterministas
src/cad_ai/pipeline.py    Orquestación mínima del vertical slice
src/cad_ai/revision.py    Patch determinista, comparación y preservación
src/cad_ai/planning.py    Contexto M3, planners, trust boundary y backends fake/HTTP
src/cad_ai/prompt_grounding.py  Evidencia lexical y validación Facts ↔ Prompt
src/cad_ai/benchmark/     Harness y fixtures RP01–RP07
src/cad_ai/demo.py        CLI reproducible
src/cad_ai/lab.py         Runner reproducible de LAB-001
tests/                    Contratos, CAD, validación e integración
```

## Autoridad de cada contrato

- `DesignSpec`: qué debe cumplir el diseño.
- `CADPlan`: cómo se intenta construir, con parámetros ya resueltos.
- `CADResult`: qué geometría produjo realmente el motor.
- `ValidationReport`: si las mediciones reales cumplen la especificación.

STEP/STL son artifacts derivados; el `model_handle` OpenCascade en memoria es la fuente geométrica durante la ejecución.

## Unidades y tolerancias

Longitudes en mm, ángulos en grados, áreas en mm² y volúmenes en mm³. La tolerancia de cada constraint es una tolerancia de diseño. No se confunde con la tolerancia numérica del kernel ni con una tolerancia de fabricación (esta última aún no se evalúa en V0.1).

## Limitaciones V0.1

- Solo tres operaciones controladas y cuerpos simples de una pieza.
- Los agujeros se ejecutan paralelos a Z y se localizan mediante ejes semánticos conocidos.
- `ANGLE` y `RADIUS` existen en el contrato pero devuelven `UNSUPPORTED`.
- Los datums proceden de la ejecución controlada; no existe reconocimiento universal ni topological naming general.
- El historial se expresa con IDs `R01/R02/...`, pero todavía no se persiste.
- El reparador solo admite constraints dimensionales `HARD + EQ`, target numérico y mapping explícito a un parámetro de una operación existente.
- Si existe cualquier `HARD FAIL` no soportado, no aplica cambios parciales ni especulativos.
- M2 no admite insertar/eliminar operaciones, cambiar el DesignSpec, ejecutar código ni corregir targets sin una regla de permiso explícita.
- M3-A no contiene un modelo real, pesos, benchmark de GPU/VRAM ni selección de runtime/modelo ganador.
- El adapter HTTP está probado con transporte simulado; todavía no se ha validado contra un `llama-server` real.
- El planner solo puede proponer `SET_PARAMETER`; no puede generar CadQuery, invocar herramientas CAD ni ejecutar bucles autónomos.
- Los retries se limitan a fallos de backend/output anteriores a la autorización semántica; un `RepairPlan` rechazado no dispara reparación alternativa automática.
