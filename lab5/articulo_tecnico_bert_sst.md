# Ajuste fino de BERT para análisis de sentimientos en SST: estudio experimental y discusión de generalización

## Resumen

El análisis de sentimientos constituye una tarea fundamental del Procesamiento del Lenguaje Natural (PLN), con aplicaciones directas en minería de opiniones, análisis de reputación y monitorización de contenido en redes sociales. En este trabajo se presenta un estudio práctico de ajuste fino (*fine-tuning*) de BERT para clasificación binaria de sentimiento sobre el corpus Stanford Sentiment Treebank (SST). El enfoque combina un codificador preentrenado de tipo Transformer con una cabeza de clasificación supervisada, aprovechando aprendizaje por transferencia para reducir la dependencia de grandes volúmenes de datos etiquetados.

Se parte de una configuración estándar basada en tokenización WordPiece y en la codificación de entradas mediante `input_ids`, `attention_mask` y `token_type_ids`. El entrenamiento se realiza con Adam (learning rate = `3e-5`, epsilon = `1e-8`, clipnorm = `1.0`), función de pérdida `SparseCategoricalCrossentropy(from_logits=True)` y métrica de exactitud *sparse*. Se analizan tanto la evolución de *loss* y *accuracy* en entrenamiento/validación como el rendimiento final en test.

Los resultados muestran que el modelo alcanza un desempeño sólido en el conjunto de prueba (test accuracy = `0.9165`; test loss = `0.3210`), lo que confirma la eficacia de BERT para esta tarea. No obstante, también se observan señales de sobreajuste incipiente: mientras la *accuracy* en entrenamiento mejora de forma consistente, la *val_loss* aumenta en épocas tardías. Este patrón sugiere que la capacidad de decisión se mantiene, pero la calibración probabilística empeora. Finalmente, se discuten implicaciones prácticas y mejoras futuras, incluyendo *early stopping*, regularización y búsqueda sistemática de hiperparámetros.

**Palabras clave:** BERT, análisis de sentimientos, fine-tuning, aprendizaje por transferencia, Transformers, SST.

---

## 1. Introducción

El crecimiento del contenido textual generado por usuarios ha convertido el análisis automático de opinión en una necesidad técnica y estratégica. Empresas, instituciones y plataformas digitales requieren sistemas capaces de identificar rápidamente la polaridad emocional de grandes volúmenes de texto para apoyar decisiones de negocio, gestión de crisis y evaluación de satisfacción de usuarios.

En este contexto, los modelos preentrenados basados en Transformers han cambiado el paradigma del PLN. Frente a enfoques clásicos (por ejemplo, modelos lineales con *n-grams* o arquitecturas recurrentes entrenadas desde cero), BERT aporta representaciones lingüísticas profundas aprendidas a gran escala, que pueden adaptarse a tareas concretas con relativamente pocos datos etiquetados. Esta adaptación, conocida como *fine-tuning*, suele mejorar la precisión y acelerar el desarrollo de soluciones prácticas.

El objetivo de este trabajo es presentar una implementación completa de ajuste fino de BERT para clasificación binaria de sentimientos sobre SST, analizar su comportamiento durante el entrenamiento y evaluar su generalización en test. Más allá de reportar métricas, se busca interpretar el comportamiento de las curvas de aprendizaje para identificar posibles limitaciones del sistema y proponer mejoras realistas.

Las contribuciones principales del estudio son:

1. Definir un *pipeline* reproducible para fine-tuning de BERT en análisis de sentimientos.
2. Evaluar experimentalmente el modelo con particiones de entrenamiento, validación y prueba.
3. Discutir señales de sobreajuste y su impacto en la interpretación de resultados.
4. Proponer líneas de mejora para escenarios académicos y aplicados.

---

## 2. Marco teórico

### 2.1 Aprendizaje por transferencia en PLN

El aprendizaje por transferencia permite reutilizar conocimiento adquirido en una tarea amplia (modelado lingüístico general) para resolver otra más específica (clasificación de sentimiento). En práctica, esto reduce la necesidad de entrenar modelos complejos desde cero y mejora el rendimiento cuando el conjunto etiquetado de la tarea objetivo es limitado.

### 2.2 BERT como codificador contextual

BERT (*Bidirectional Encoder Representations from Transformers*) utiliza autoatención bidireccional para construir representaciones contextualizadas de cada token. Gracias al preentrenamiento en grandes corpus, codifica patrones sintácticos y semánticos útiles para múltiples tareas posteriores.

En clasificación de secuencias, una estrategia habitual consiste en tomar la representación del token especial `[CLS]` y conectarla a una capa densa para producir logits de clase. Esta decisión arquitectónica, aunque simple, suele ser suficiente para obtener resultados competitivos en tareas de polaridad textual.

### 2.3 Métricas y comportamiento de entrenamiento

En tareas de clasificación, la *accuracy* indica proporción de aciertos, mientras la función de pérdida (entropía cruzada) refleja además la confianza de las predicciones. Por ello, pueden aparecer escenarios donde la *accuracy* se mantiene estable y, sin embargo, la *loss* empeora, señalando descalibración o inestabilidad en las probabilidades estimadas.

---

## 3. Datos y definición de la tarea

Se utiliza Stanford Sentiment Treebank (SST) en una configuración de clasificación binaria. El conjunto original con escala de sentimiento se transforma a dos clases (negativa/positiva), descartando el valor neutro para simplificar la tarea y hacerla consistente con un enfoque binario.

Tamaños de partición utilizados:

- **Entrenamiento:** 6920 ejemplos
- **Validación:** 872 ejemplos
- **Prueba:** 1821 ejemplos

Las muestras se barajan (*shuffle*) para reducir sesgos de orden y se fijan semillas aleatorias para mejorar reproducibilidad experimental.

La tarea se formula como:

`f(x) -> y`, con `y in {0, 1}`,

donde `x` es una oración y `y` representa polaridad negativa (`0`) o positiva (`1`).

---

## 4. Metodología

### 4.1 Preprocesamiento y tokenización

El texto se procesa con el tokenizador asociado al modelo BERT elegido, basado en WordPiece. Este tokenizador divide palabras en subunidades cuando es necesario, lo cual mitiga el problema de vocabulario cerrado y permite manejar palabras raras o morfológicamente complejas.

Para cada ejemplo textual se generan:

- `input_ids`: índices de tokens en el vocabulario.
- `attention_mask`: máscara para distinguir tokens reales de *padding*.
- `token_type_ids`: identificadores de segmento (útiles en entradas dobles; aquí se mantienen por compatibilidad con la arquitectura).

Posteriormente, las características se convierten a un `tf.data.Dataset` para entrenamiento eficiente por lotes.

### 4.2 Arquitectura del modelo

El clasificador final integra dos componentes:

1. **Encoder BERT preentrenado**: extrae representaciones contextuales.
2. **Cabeza de clasificación**:
   - Entrada: `pooler_output` (representación asociada a `[CLS]`).
   - Regularización: `Dropout(0.1)`.
   - Salida: `Dense(2)` para logits de dos clases.

Esta arquitectura mantiene un equilibrio entre capacidad de representación (aportada por BERT) y simplicidad de ajuste para la tarea específica.

### 4.3 Configuración de optimización

Se emplea la siguiente configuración:

- **Optimizador:** Adam
  - learning rate = `3e-5`
  - epsilon = `1e-8`
  - clipnorm = `1.0`
- **Pérdida:** `SparseCategoricalCrossentropy(from_logits=True)`
- **Métrica principal:** `SparseCategoricalAccuracy`
- **Número de épocas:** 3

La selección de parámetros sigue prácticas comunes de fine-tuning en Transformers y favorece una comparación estable entre variantes.

---

## 5. Diseño experimental

El procedimiento de evaluación se organiza en tres fases:

1. **Entrenamiento** con seguimiento de `loss` y `accuracy` en entrenamiento/validación.
2. **Inspección de curvas** para detectar convergencia, estancamiento o sobreajuste.
3. **Evaluación final en test** con los mismos mecanismos de tokenización y preparación usados en entrenamiento.

Este esquema permite distinguir entre desempeño aprendido y capacidad real de generalización fuera de muestra.

---

## 6. Resultados

### 6.1 Evolución durante el entrenamiento

En una de las ejecuciones principales se observa:

- Época 1: train accuracy ~ `0.8455`, val accuracy ~ `0.9128`, val loss ~ `0.2235`
- Época 2: train accuracy ~ `0.9509`, val accuracy ~ `0.9094`, val loss ~ `0.2649`
- Época 3: train accuracy ~ `0.9822`, val accuracy ~ `0.9083`, val loss ~ `0.3882`

En otra variante reportada en el mismo trabajo, la `val_accuracy` mejora ligeramente en épocas posteriores mientras `val_loss` también aumenta. Ambos comportamientos son compatibles con un fenómeno de sobreajuste incipiente centrado en calibración de confianza más que en frontera de decisión.

### 6.2 Rendimiento en test

Resultado final sobre el conjunto de prueba:

- **Test loss:** `0.3210`
- **Test accuracy:** `0.9165`

Este valor de exactitud confirma que el sistema alcanza un rendimiento alto para una configuración académica estándar, mostrando la efectividad del ajuste fino de BERT en análisis de sentimientos binario.

---

## 7. Discusión

### 7.1 Interpretación del patrón `val_loss` vs `val_accuracy`

El aspecto más relevante del experimento no es solo la cifra final en test, sino el desacople parcial entre métricas en validación:

- La *train accuracy* crece con fuerza.
- La *val_accuracy* se mantiene estable (o sube levemente según la variante).
- La *val_loss* aumenta hacia épocas tardías.

Este patrón sugiere que el modelo sigue clasificando correctamente una proporción alta de ejemplos, pero con probabilidades menos bien calibradas (predicciones más extremas o menos robustas). En términos prácticos, puede implicar mayor sensibilidad a cambios de dominio o a ejemplos ambiguos, aunque la exactitud agregada permanezca alta.

### 7.2 Evidencia de sobreajuste

Sí hay indicios de **sobreajuste incipiente**, no necesariamente grave pero visible. No se observa colapso de `val_accuracy`, por lo que la generalización global sigue siendo buena; sin embargo, el crecimiento de `val_loss` aconseja precaución al aumentar épocas sin controles adicionales.

### 7.3 Robustez de la evaluación

Con `test accuracy = 0.9165`, el modelo demuestra solidez para la tarea planteada. Aun así, un análisis más completo podría incluir:

- *Precision*, *recall* y F1 por clase.
- Matriz de confusión.
- Evaluación de calibración (por ejemplo, ECE o curvas de confiabilidad).
- Variabilidad entre semillas.

Esto permitiría caracterizar mejor el comportamiento en casos límite y reducir el riesgo de conclusiones optimistas basadas en una sola métrica.

---

## 8. Limitaciones

1. **Métrica principal limitada:** se prioriza *accuracy*; faltan métricas por clase.
2. **Número de épocas fijo:** no se aplica *early stopping* guiado por validación.
3. **Exploración de hiperparámetros acotada:** no hay búsqueda sistemática.
4. **Sin análisis cualitativo de errores:** no se inspeccionan ejemplos mal clasificados.
5. **Dominio único:** resultados sobre SST no garantizan transferencia directa a otros dominios.

---

## 9. Trabajo futuro

Para mejorar el sistema y la calidad del análisis, se proponen las siguientes líneas:

- Implementar **early stopping** monitorizando `val_loss`.
- Realizar **búsqueda de hiperparámetros** (learning rate, dropout, batch size).
- Incorporar **weight decay** y/o estrategias de regularización adicionales.
- Reportar **F1 macro**, **matriz de confusión** y métricas de calibración.
- Comparar con variantes como **DistilBERT** o **RoBERTa**.
- Añadir análisis de errores por fenómeno lingüístico (negación, ironía, ambigüedad léxica).

---

## 10. Conclusiones

Este trabajo confirma que el ajuste fino de BERT es una estrategia eficaz para análisis de sentimientos binario en SST. El modelo alcanza una exactitud en test de `0.9165` con una arquitectura relativamente simple y una configuración estándar de entrenamiento, lo que evidencia el valor práctico del aprendizaje por transferencia en PLN.

Al mismo tiempo, el comportamiento de las curvas de validación muestra que no basta con observar *accuracy*: la evolución de *loss* aporta información crítica sobre calibración y riesgo de sobreajuste. Por tanto, una evaluación rigurosa debe combinar métricas complementarias y criterios de parada más robustos.

En conjunto, los resultados respaldan BERT como base sólida para tareas de opinión y señalan un camino claro para mejorar robustez, interpretabilidad y capacidad de generalización en escenarios más exigentes.

---

## Referencias

- Devlin, J., Chang, M.-W., Lee, K., y Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Socher, R., et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. EMNLP.
- Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP: System Demonstrations.
