"""Construye aprendizaje por refuerzo, transformer/LLM y plantilla del proyecto."""

from __future__ import annotations

from build_classical_notebooks import header
from build_initial_notebooks import code, md, write_notebook


def build_reinforcement_learning() -> None:
    path = "07_temas_actuales/70_aprendizaje_por_refuerzo.ipynb"
    cells = [
        header(path, "Aprendizaje por refuerzo: Q-learning", "¿Cómo aprende un agente de consecuencias demoradas?", 4),
        md(
            r"""
            ## Proceso de decisión de Markov

            Un MDP contiene estados $s$, acciones $a$, transición, recompensa $r$ y
            descuento $\gamma$. La función óptima satisface

            $$Q^*(s,a)=E\left[r+\gamma\max_{a'}Q^*(s',a')\right].$$

            Q-learning aproxima esa ecuación con
            $Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]$.
            Es *off-policy*: el objetivo usa la acción codiciosa aunque la conducta
            explore con $\varepsilon$-greedy.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np

            FILAS,COLUMNAS=6,6
            INICIO=(5,0); META=(0,5); OBSTÁCULOS={(1,1),(1,2),(2,2),(3,2),(4,4)}
            ACCIONES=[(-1,0),(1,0),(0,-1),(0,1)]
            FLECHAS=np.array(["↑","↓","←","→"])

            def paso(estado,acción):
                dr,dc=ACCIONES[acción]; candidato=(estado[0]+dr,estado[1]+dc)
                fuera=not(0<=candidato[0]<FILAS and 0<=candidato[1]<COLUMNAS)
                siguiente=estado if fuera or candidato in OBSTÁCULOS else candidato
                recompensa=10.0 if siguiente==META else -1.0
                return siguiente,recompensa,siguiente==META

            def índice(estado): return estado[0]*COLUMNAS+estado[1]
            """
        ),
        code(
            """
            SEMILLA=42; rng=np.random.default_rng(SEMILLA)
            Q=np.zeros((FILAS*COLUMNAS,len(ACCIONES))); retornos=[]
            alpha=.15; gamma=.97
            for episodio in range(2500):
                estado=INICIO; retorno=0; epsilon=max(.03,1-episodio/1800)
                for _ in range(250):
                    if rng.random()<epsilon: acción=rng.integers(len(ACCIONES))
                    else: acción=np.argmax(Q[índice(estado)])
                    siguiente,r,fin=paso(estado,acción)
                    objetivo=r if fin else r+gamma*np.max(Q[índice(siguiente)])
                    Q[índice(estado),acción]+=alpha*(objetivo-Q[índice(estado),acción])
                    estado=siguiente; retorno+=r
                    if fin: break
                retornos.append(retorno)

            media=np.convolve(retornos,np.ones(100)/100,mode="valid")
            plt.plot(media); plt.xlabel("episodio"); plt.ylabel("retorno medio (100)"); plt.show()
            """
        ),
        code(
            """
            política=np.full((FILAS,COLUMNAS)," ",dtype=object)
            valor=np.max(Q,axis=1).reshape(FILAS,COLUMNAS)
            for f in range(FILAS):
                for c in range(COLUMNAS):
                    if (f,c) in OBSTÁCULOS: política[f,c]="■"
                    elif (f,c)==META: política[f,c]="★"
                    else: política[f,c]=FLECHAS[np.argmax(Q[índice((f,c))])]
            print(política)
            plt.imshow(valor,cmap="viridis"); plt.colorbar(label="max Q"); plt.title("Función de valor aprendida"); plt.show()

            estado=INICIO; ruta=[estado]
            for _ in range(50):
                estado,_,fin=paso(estado,np.argmax(Q[índice(estado)])); ruta.append(estado)
                if fin: break
            print("ruta:",ruta,"pasos:",len(ruta)-1)
            """
        ),
        md(
            """
            La recompensa define lo que el agente optimiza, no necesariamente lo que
            queríamos. Este entorno conoce todos sus estados; redes profundas se
            vuelven útiles cuando el estado es grande o continuo.

            **Ejercicios:** quite descuento; haga obstáculos estocásticos; compare
            SARSA; diseñe una recompensa que produzca una conducta indeseada; reporte
            media e intervalo de retornos en 20 semillas, no sólo la mejor corrida.
            """
        ),
    ]
    write_notebook(path, cells)


def build_transformers_llm() -> None:
    path = "07_temas_actuales/71_transformers_y_llm.ipynb"
    cells = [
        header(path, "Transformers, GPT y modelos de lenguaje", "¿Cómo predice el siguiente token un transformer causal?", 6),
        md(
            r"""
            **Requiere PyTorch; GPU opcional.** Este notebook construye un GPT
            diminuto de caracteres. Busca revelar los componentes, no reproducir la
            escala ni las capacidades de un sistema comercial.

            Flujo conceptual:

            1. tokenizar texto y convertir tokens en embeddings;
            2. mezclar contexto con autoatención causal;
            3. preentrenar mediante predicción del siguiente token;
            4. en sistemas asistentes, realizar postentrenamiento para seguir
               instrucciones y preferencias;
            5. durante inferencia, muestrear tokens autoregresivamente y, en algunos
               sistemas, coordinar herramientas externas.
            """
        ),
        md(
            r"""
            ## Atención escalada

            Desde una secuencia $X$ calculamos $Q=XW_Q$, $K=XW_K$, $V=XW_V$:

            $$\mathrm{Attention}(Q,K,V)=
            \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V.$$

            La máscara causal $M_{ij}=-\infty$ si $j>i$: la posición $i$ no puede
            mirar tokens futuros. Varias cabezas aprenden proyecciones distintas y
            concatenan sus resultados. La atención por sí sola no conoce el orden;
            añadimos embeddings posicionales.
            """
        ),
        code(
            """
            import math
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn.functional as F
            from torch import nn

            def softmax_estable(x,axis=-1):
                e=np.exp(x-np.max(x,axis=axis,keepdims=True)); return e/e.sum(axis=axis,keepdims=True)

            rng=np.random.default_rng(42); X_demo=rng.normal(size=(5,4))
            Q=X_demo@rng.normal(size=(4,4)); K=X_demo@rng.normal(size=(4,4)); V=X_demo@rng.normal(size=(4,4))
            scores=Q@K.T/np.sqrt(4); scores[np.triu_indices(5,k=1)]=-np.inf
            A=softmax_estable(scores); contexto=A@V
            print("filas suman",A.sum(axis=1)); print("contexto",contexto.shape)
            plt.imshow(A,cmap="viridis",vmin=0,vmax=1); plt.xlabel("token consultado"); plt.ylabel("token que consulta"); plt.colorbar(); plt.show()
            """
        ),
        md(
            r"""
            ## Objetivo autoregresivo

            Dada una secuencia $x_1,\ldots,x_T$, maximizamos
            $\sum_t\log p_\theta(x_t\mid x_{<t})$. La entropía cruzada compara los
            logits del vocabulario con el siguiente token. *Teacher forcing* permite
            calcular todas las posiciones en paralelo durante entrenamiento; generar
            sigue siendo secuencial.

            El corpus siguiente es deliberadamente pequeño y escrito para esta
            práctica. Un modelo real necesita mucha más diversidad, cómputo,
            evaluación, documentación de datos y salvaguardas.
            """
        ),
        code(
            """
            SEMILLA=42; torch.manual_seed(SEMILLA)
            dispositivo=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            base=("la energia se conserva en un sistema aislado. "
                  "la fuerza cambia el momento. la luz transporta energia y momento. "
                  "un modelo aproxima datos dentro de supuestos. medir implica incertidumbre.\\n")
            texto=base*220
            caracteres=sorted(set(texto)); stoi={c:i for i,c in enumerate(caracteres)}; itos={i:c for c,i in stoi.items()}
            datos=torch.tensor([stoi[c] for c in texto],dtype=torch.long)
            corte=int(.9*len(datos)); datos_train,datos_val=datos[:corte],datos[corte:]
            VOCAB=len(caracteres); BLOQUE=48; BATCH=64

            def lote(fuente):
                i=torch.randint(len(fuente)-BLOQUE-1,(BATCH,))
                x=torch.stack([fuente[j:j+BLOQUE] for j in i]); y=torch.stack([fuente[j+1:j+BLOQUE+1] for j in i])
                return x.to(dispositivo),y.to(dispositivo)
            print("vocabulario",VOCAB,"tokens",len(datos),"dispositivo",dispositivo)
            """
        ),
        code(
            """
            class AtenciónCausal(nn.Module):
                def __init__(self,dimensión=64,cabezas=4):
                    super().__init__(); self.cabezas=cabezas; self.dk=dimensión//cabezas
                    self.qkv=nn.Linear(dimensión,3*dimensión); self.proyección=nn.Linear(dimensión,dimensión)
                    self.register_buffer("máscara",torch.tril(torch.ones(BLOQUE,BLOQUE,dtype=torch.bool)))
                def forward(self,x):
                    B,T,C=x.shape
                    qkv=self.qkv(x).reshape(B,T,3,self.cabezas,self.dk).permute(2,0,3,1,4)
                    q,k,v=qkv[0],qkv[1],qkv[2]
                    pesos=(q@k.transpose(-2,-1))/math.sqrt(self.dk)
                    pesos=pesos.masked_fill(~self.máscara[:T,:T],float("-inf")); pesos=torch.softmax(pesos,dim=-1)
                    salida=(pesos@v).transpose(1,2).contiguous().reshape(B,T,C)
                    return self.proyección(salida)

            class BloqueTransformer(nn.Module):
                def __init__(self,dimensión=64,cabezas=4):
                    super().__init__(); self.ln1=nn.LayerNorm(dimensión); self.ln2=nn.LayerNorm(dimensión)
                    self.atención=AtenciónCausal(dimensión,cabezas)
                    self.mlp=nn.Sequential(nn.Linear(dimensión,4*dimensión),nn.GELU(),nn.Linear(4*dimensión,dimensión))
                def forward(self,x):
                    x=x+self.atención(self.ln1(x)); return x+self.mlp(self.ln2(x))

            class MiniGPT(nn.Module):
                def __init__(self,dimensión=64,capas=2,cabezas=4):
                    super().__init__(); self.token=nn.Embedding(VOCAB,dimensión); self.pos=nn.Embedding(BLOQUE,dimensión)
                    self.bloques=nn.Sequential(*[BloqueTransformer(dimensión,cabezas) for _ in range(capas)])
                    self.ln=nn.LayerNorm(dimensión); self.cabeza=nn.Linear(dimensión,VOCAB,bias=False)
                def forward(self,idx,objetivo=None):
                    T=idx.shape[1]; x=self.token(idx)+self.pos(torch.arange(T,device=idx.device))
                    logits=self.cabeza(self.ln(self.bloques(x)))
                    pérdida=None if objetivo is None else F.cross_entropy(logits.reshape(-1,VOCAB),objetivo.reshape(-1))
                    return logits,pérdida
                @torch.no_grad()
                def generar(self,idx,nuevos=180,temperatura=.8):
                    for _ in range(nuevos):
                        logits,_=self(idx[:,-BLOQUE:]); prob=torch.softmax(logits[:,-1]/temperatura,dim=-1)
                        idx=torch.cat([idx,torch.multinomial(prob,1)],dim=1)
                    return idx
            """
        ),
        code(
            """
            modelo=MiniGPT().to(dispositivo); optimizador=torch.optim.AdamW(modelo.parameters(),lr=3e-3,weight_decay=.01)
            print("parámetros",sum(p.numel() for p in modelo.parameters()))
            historial=[]
            for paso in range(700):
                xb,yb=lote(datos_train); optimizador.zero_grad(); _,pérdida=modelo(xb,yb); pérdida.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(),1.0); optimizador.step()
                if paso%100==0:
                    modelo.eval()
                    with torch.no_grad(): xv,yv=lote(datos_val); val=modelo(xv,yv)[1].item()
                    modelo.train(); historial.append((paso,pérdida.item(),val)); print(historial[-1])
            """
        ),
        code(
            """
            modelo.eval()
            inicio=torch.tensor([[stoi["l"]]],device=dispositivo)
            generado=modelo.generar(inicio,nuevos=240,temperatura=.75)[0].cpu().tolist()
            print("".join(itos[i] for i in generado))
            h=np.asarray(historial); plt.plot(h[:,0],h[:,1],label="train"); plt.plot(h[:,0],h[:,2],label="validation"); plt.xlabel("paso"); plt.ylabel("cross entropy"); plt.legend(); plt.show()
            """
        ),
        md(
            """
            ## Qué explica esto sobre asistentes actuales

            El experimento sí ilustra tokenización, embeddings, atención causal,
            bloques residuales, predicción autoregresiva, temperatura y
            preentrenamiento. Sistemas actuales pueden además aceptar varias
            modalidades, usar ventanas de contexto grandes, razonamiento especializado
            y herramientas; las capacidades publicadas deben consultarse en la
            [documentación oficial de modelos de OpenAI](https://developers.openai.com/api/docs/models).

            No se deben inventar detalles sobre un producto concreto. El número de
            parámetros, mezcla exacta de datos, arquitectura completa, postentrenamiento,
            infraestructura de inferencia y reglas internas pueden no ser públicos.
            Una explicación responsable separa el mecanismo académico conocido de
            aquello que la organización realmente documenta.

            Lecturas primarias: [Transformer](https://arxiv.org/abs/1706.03762),
            [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf),
            [GPT-3](https://arxiv.org/abs/2005.14165) e
            [InstructGPT](https://arxiv.org/abs/2203.02155).
            """
        ),
        md(
            """
            **Ejercicios:** cambie longitud de contexto y mida validación; elimine
            posiciones; visualice una cabeza; compare greedy, temperatura y top-k;
            calcule costo $O(T^2)$ de la matriz de atención; documente sesgos del
            corpus y proponga evaluaciones de factualidad, seguridad y memorization.
            """
        ),
    ]
    write_notebook(path, cells, tier="pytorch", accelerator="gpu-optional")


def build_project_template() -> None:
    path = "08_proyecto/80_plantilla_proyecto_final.ipynb"
    cells = [
        header(path, "Plantilla reproducible del proyecto final", "¿Puede otra persona reconstruir y cuestionar el resultado?", 2),
        md(
            """
            ## Identificación

            - **Título:** complete aquí.
            - **Integrantes:** tres estudiantes.
            - **Modelo neural no estudiado directamente en clase:** complete aquí.
            - **Pregunta física y predicción falsable:** complete aquí.
            - **Sustentación:** dentro de la franja concertada del 10 al 18 de noviembre.

            El componente escrito representa 30 % y la sustentación 20 % de la nota
            del curso según el acta suministrada. Confirme cualquier ajuste anunciado
            posteriormente por el docente.
            """
        ),
        md(
            r"""
            ## 1. Fundamento matemático

            Defina entradas, salidas, arquitectura, operación forward, pérdida,
            regularización y regla de actualización. Incluya dimensiones de tensores
            y número de parámetros. Explique por qué las simetrías o invariancias del
            método tienen sentido para el sistema físico.

            **TODO del grupo:** escriba aquí las ecuaciones con notación consistente.
            """
        ),
        md(
            """
            ## 2. Datos y partición

            Documente URL estable, licencia, versión/fecha, variables, unidades,
            criterios de exclusión y posibles sesgos. La unidad de partición debe ser
            el objeto físico o experimento, no una ventana/imagen derivada, para evitar
            que réplicas del mismo fenómeno crucen train y test.
            """
        ),
        code(
            """
            from dataclasses import asdict, dataclass
            import json, platform
            import numpy as np
            import pandas as pd

            @dataclass(frozen=True)
            class Configuración:
                semilla: int = 42
                fracción_test: float = 0.20
                fracción_validation: float = 0.20
                épocas_máximas: int = 100
                paciencia: int = 10
                métrica_primaria: str = "DEFINIR"

            CONFIG=Configuración()
            np.random.seed(CONFIG.semilla)
            print(json.dumps({**asdict(CONFIG),"python":platform.python_version()},indent=2))

            # TODO: cargar datos desde una fuente documentada y comprobar esquema.
            # datos = ...
            """
        ),
        md(
            """
            ## 3. Línea base, modelo y selección

            Incluya una línea base ingenua y una competitiva. El modelo complejo sólo
            tiene valor si supera alternativas simples bajo la misma partición y
            métrica. Seleccione hiperparámetros con validation/CV; abra el test una vez.
            """
        ),
        code(
            """
            columnas_resultado=["modelo","parámetros_entrenables","tiempo_s","métrica_validation","métrica_test","semilla","observaciones"]
            resultados=pd.DataFrame(columns=columnas_resultado)
            display(resultados)

            def registrar_resultado(tabla, **fila):
                faltantes=set(columnas_resultado)-set(fila)
                if faltantes: raise ValueError(f"Faltan campos: {sorted(faltantes)}")
                return pd.concat([tabla,pd.DataFrame([fila])],ignore_index=True)
            """
        ),
        md(
            """
            ## 4. Evidencia mínima

            - curva train/validation y criterio de parada;
            - métrica primaria con incertidumbre o repetición de semillas;
            - matriz de confusión o residuos según el problema;
            - ablación de al menos dos decisiones arquitectónicas;
            - análisis de errores con ejemplos concretos;
            - costo computacional y huella aproximada;
            - límites de generalización y riesgos de fuga;
            - enlace a código, entorno y artefactos necesarios.

            ## 5. Estructura sugerida de la sustentación

            1. problema físico y datos;
            2. fundamento matemático del método nuevo;
            3. diseño experimental y línea base;
            4. resultados, incertidumbre y ablaciones;
            5. fallo más interesante;
            6. conclusión y siguiente experimento.
            """
        ),
        code(
            """
            lista_control={
                "test_intacto":False,
                "pipeline_sin_fuga":False,
                "línea_base":False,
                "matemática_y_formas":False,
                "ablaciones":False,
                "varias_semillas":False,
                "licencia_datos":False,
                "notebook_desde_cero":False,
            }
            display(pd.Series(lista_control,name="completado"))
            print("Elementos pendientes:",[k for k,v in lista_control.items() if not v])
            """
        ),
        md(
            """
            Antes de entregar, reinicie el runtime y ejecute **Run all**. El notebook
            final debe funcionar sin archivos privados ni rutas del computador del
            estudiante. Las afirmaciones deben corresponder a las figuras y tablas
            realmente obtenidas, no sólo a expectativas teóricas.
            """
        ),
    ]
    write_notebook(path, cells)


if __name__ == "__main__":
    build_reinforcement_learning()
    build_transformers_llm()
    build_project_template()
    print("Notebooks de temas actuales y proyecto generados.")
