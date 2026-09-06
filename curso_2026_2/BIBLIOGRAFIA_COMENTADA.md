# Bibliografía comentada para una futura edición

Consulta actualizada: 6 de septiembre de 2026.

## Recomendación principal

**Simon J. D. Prince, _Understanding Deep Learning_ (MIT Press, 2023).**

Lo usaría como texto troncal en una futura versión del curso. Conserva el estilo
que interesa para estudiantes de Física: comienza con intuición, formula cada
idea matemáticamente y la acompaña con visualizaciones e implementaciones
pequeñas. Su edición abierta incluye ejercicios en notebooks. El libro incorpora
transformers y modelos de difusión; el sitio oficial añade material actualizado
sobre LLM, preentrenamiento, ajuste por instrucciones, RLHF, explicabilidad y
privacidad diferencial.

- [Página editorial y edición abierta](https://mitpress.mit.edu/9780262377102/understanding-deep-learning/)
- [Libro, notebooks y extensiones del autor](https://udlbook.github.io/udlbook/)

### ¿Reemplaza a Calin?

**Como secuencia docente inicial, sí es una mejora; como fuente matemática, no lo
reemplaza por completo.**

| Uso | Prince | Calin |
|---|---|---|
| Primera lectura del estudiante | Principal | Consulta selectiva |
| Matemática aplicada y visual | Muy fuerte | Fuerte |
| Teoría de aproximación, geometría y análisis más profundo | Selectiva | Principal |
| Transformers, difusión y extensión a LLM | Actual | No cubre la era moderna |
| Notebooks listos para adaptar | Sí | No es su centro |

La combinación propuesta es: **Prince organiza el curso; Calin profundiza las
derivaciones que merecen tratamiento matemático adicional**. No conviene obligar
a leer ambos linealmente.

## Complemento específico para GPT

**Sebastian Raschka, _Build a Large Language Model (From Scratch)_ (Manning,
2024).** Es el mejor complemento práctico para el notebook final: construye un
transformer tipo GPT sin esconder sus piezas en una biblioteca de LLM, carga
pesos preentrenados y recorre preentrenamiento, clasificación y ajuste para
seguir instrucciones. Requiere Python intermedio y conocimientos previos de ML,
por lo que **no sustituye** ni a Prince ni a Calin desde el comienzo.

- [Descripción, alcance y recursos oficiales](https://www.manning.com/books/build-a-large-language-model-from-scratch)

Para este curso condensado tomaría de Raschka sólo el recorrido conceptual y un
GPT diminuto en Colab; entrenar un modelo grande no es un objetivo razonable ni
necesario.

## Alternativa con continuidad Springer

**Christopher M. Bishop y Hugh Bishop, _Deep Learning: Foundations and
Concepts_ (Springer, 2024).** Es una alternativa excelente si se prefiere un
tratado lineal y autocontenido: incluye probabilidad, redes, gradiente,
retropropagación, regularización, CNN, transformers, GNN y modelos generativos.
Su capítulo de transformers es moderno, pero para programar un GPT completo
seguiría siendo necesario el complemento de Raschka.

- [Ficha y tabla de contenido oficiales](https://link.springer.com/book/10.1007/978-3-031-45468-4)
