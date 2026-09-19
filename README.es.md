<p align="center">
  <a href="./README.md">🇬🇧 English</a> · <strong>🇪🇸 Español</strong>
</p>

<h1 align="center">CAD AI</h1>

<p align="center">
  Un sistema local-first para transformar requisitos en lenguaje natural en
  <strong>CAD paramétrico validado</strong> mediante grounding determinista,
  planificación estructurada y reparación automática controlada.
</p>

<p align="center">
  <strong>Lenguaje natural → hechos grounded → DesignSpec → CADPlan → geometría → validación → reparación → STEP/STL</strong>
</p>

---

## Descripción

CAD AI explora una idea sencilla: un LLM puede ayudar a interpretar la intención de diseño, pero **no debe ser la autoridad que crea o aprueba directamente la geometría**.

El sistema separa interpretación y ejecución.

Los requisitos en lenguaje natural se convierten en hechos estructurados, pasan por gates deterministas, se transforman en una especificación formal y un plan CAD, se ejecutan mediante un motor paramétrico y finalmente son comprobados por un validador independiente. Si la validación falla, un bucle de reparación restringido puede proponer una corrección sin entregar al LLM control libre sobre la geometría.

> **Estado:** desarrollo activo. La arquitectura actual del prototipo es funcional y está cubierta por tests automatizados y casos de benchmark. El repositorio público se está preparando para una importación limpia del código.

## Pipeline principal

```text
User Prompt
    │
    ▼
DeterministicPromptGrounder
    │
    ▼
ExtractionCoverage
    │
    ├── COMPLETE ─────────► DeterministicFactAssembler
    │
    ├── PARTIAL ──────────► ResidualLLMExtractor
    │                           │
    │                           ▼
    │                      ResidualFacts
    │                           │
    │                           ▼
    │                  DeterministicFactAssembler
    │                           │
    │                           ▼
    │                  FactGroundingValidator
    │
    └── INSUFFICIENT ─────► Error estructurado / no CAD

Canonical Extracted Facts
    │
    ▼
Deterministic Intent Gate
    │
    ▼
Intent
    │
    ▼
DesignSpec
    │
    ▼
CADPlan
    │
    ▼
CAD Engine
    │
    ▼
Validator
    │
    ├── PASS ─────────────► STEP / STL
    │
    └── FAIL
         │
         ▼
    Repair Planner
         │
         ▼
    RepairPlanValidator
         │
         ▼
    Repair Executor
         │
         └───────────────► revalidación
```

Hay una descripción más detallada en [docs/architecture.md](./docs/architecture.md).

## Principios de diseño

CAD AI se apoya en reglas estrictas:

- **El LLM propone e interpreta; nunca ejecuta geometría directamente.**
- **La evidencia determinista tiene prioridad sobre la salida del modelo.**
- Los facts grounded no pueden ser sobrescritos por la extracción residual del LLM.
- **Schema-valid no significa autorizado.**
- El Intent Gate determinista mantiene la autoridad sobre qué intención está soportada.
- `DesignSpec → CADPlan` es determinista.
- La validación es independiente de la generación.
- Las reparaciones están restringidas por un validador y un ejecutor determinista.
- Si falta información se devuelve un error estructurado, no CAD especulativo.
- Los fallos importantes corregidos se convierten en tests de regresión.

Consulta [docs/principles.md](./docs/principles.md).

## Capacidades actuales

El prototipo actual se centra en generación paramétrica de piezas y modificación controlada de features.

Áreas implementadas o validadas:

- grounding determinista del prompt;
- cobertura de extracción complete / partial / insufficient;
- extracción residual mediante LLM para hechos no resueltos;
- ensamblado canónico y validación de grounding;
- intent gating determinista;
- `DesignSpec` y `CADPlan` formales;
- generación CAD con CadQuery / OpenCascade;
- validación independiente;
- ejecución determinista de reparaciones;
- planificación de reparación asistida por LLM pero restringida;
- structured output;
- experimentos de inferencia con LLMs locales;
- tests de regresión y suites de benchmark;
- generación STEP / STL.

## Ejemplo de reparación

Una revisión puede fallar una restricción dimensional sin invalidar el diseño completo.

```text
R01
 └─ Validator
      └─ FAIL: diámetro del agujero fuera de tolerancia
             │
             ▼
       Repair context
             │
             ▼
       Repair planner
             │
             ▼
       SET_PARAMETER
       target: diámetro
             │
             ▼
       RepairPlanValidator
             │
             ▼
       Ejecución determinista
             │
             ▼
            R02
             │
             ▼
        Validator PASS
```

La reparación debe ser local: parámetros y restricciones no relacionados deben conservarse.

## Tecnologías

| Área | Tecnologías |
| --- | --- |
| Lenguaje | Python |
| CAD | CadQuery · OpenCascade |
| IA | LLMs locales · structured output · JSON Schema |
| Experimentos de inferencia | llama.cpp · Vulkan |
| Validación | Checks deterministas de geometría / constraints |
| Testing | pytest · regresiones · benchmarks |
| Salidas | STEP · STL |

## Estructura del repositorio

El repositorio público se está preparando alrededor de esta estructura:

```text
CAD-AI/
├── src/             # Paquete principal
├── tests/           # Tests unitarios y de regresión
├── benchmarks/      # Casos de evaluación reproducibles
├── examples/        # Ejemplos pequeños de uso
├── docs/            # Arquitectura y documentación de diseño
├── assets/          # Diagramas y capturas
├── pyproject.toml
├── README.md
└── README.es.md
```

El código se importará solo después de revisar outputs generados, artefactos de modelos locales, resultados de tests y configuración específica de la máquina.

## Filosofía de evaluación

El proyecto distingue entre:

- **unit tests** — corrección de componentes individuales;
- **CAD benchmarks** — si la geometría cumple los requisitos esperados;
- **repair benchmarks** — si los fallos se corrigen de forma segura;
- **regression tests** — si los fallos corregidos permanecen corregidos;
- **model benchmarks** — si distintos modelos/prompts producen propuestas estructuradas válidas bajo condiciones idénticas.

Las métricas útiles incluyen validation pass rate, repair success rate, first-pass success, average revisions, constraint preservation, change locality, determinismo, latencia y uso de recursos.

## ¿Por qué local-first?

La arquitectura es deliberadamente model-agnostic.

Los modelos locales aportan:

- privacidad;
- reproducibilidad;
- experimentación offline;
- condiciones estables de benchmark;
- independencia de un único proveedor.

El sistema puede soportar también modelos remotos, pero ningún modelo se considera fuente de verdad geométrica.

## Licencia

Todavía no se ha seleccionado una licencia. Hasta que se añada una, se aplican las reglas de copyright por defecto.

---

Construido en España 🇪🇸 como proyecto personal de ingeniería e investigación en IA/CAD.
