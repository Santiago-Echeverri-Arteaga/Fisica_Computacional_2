"""Construye notebooks de CNN, transferencia, generativos y secuencias."""

from __future__ import annotations

from build_classical_notebooks import header
from build_initial_notebooks import code, md, write_notebook


def build_convolution_numpy() -> None:
    path = "04_vision/40_convolucion_desde_cero.ipynb"
    cells = [
        header(path, "Convolución y pooling desde cero", "¿Qué información local extrae un filtro?", 4),
        md(
            r"""
            ## Correlación usada por las CNN

            Las bibliotecas suelen llamar convolución a la correlación cruzada
            $$Y_{ij}=\sum_{u,v}X_{i+u,j+v}K_{uv}+b.$$
            Con entrada $H\times W$, kernel $K_h\times K_w$, padding $P$ y stride
            $S$, la altura de salida es
            $\lfloor(H+2P-K_h)/S\rfloor+1$. Los pesos compartidos detectan el mismo
            patrón en posiciones distintas.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.datasets import load_digits

            def correlación_2d(imagen, kernel, stride=1, padding=0):
                imagen = np.pad(imagen, padding)
                kh, kw = kernel.shape
                oh = (imagen.shape[0]-kh)//stride + 1
                ow = (imagen.shape[1]-kw)//stride + 1
                salida = np.empty((oh, ow), dtype=float)
                for i in range(oh):
                    for j in range(ow):
                        región = imagen[i*stride:i*stride+kh, j*stride:j*stride+kw]
                        salida[i, j] = np.sum(región*kernel)
                return salida

            def max_pool_2d(imagen, tamaño=2, stride=2):
                oh = (imagen.shape[0]-tamaño)//stride + 1
                ow = (imagen.shape[1]-tamaño)//stride + 1
                salida = np.empty((oh, ow))
                for i in range(oh):
                    for j in range(ow):
                        región = imagen[i*stride:i*stride+tamaño, j*stride:j*stride+tamaño]
                        salida[i,j] = región.max()
                return salida
            """
        ),
        code(
            """
            digits = load_digits()
            imagen = digits.images[np.flatnonzero(digits.target == 8)[0]]
            kernels = {
                "Sobel horizontal": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
                "Sobel vertical": np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
                "Laplaciano": np.array([[0,1,0],[1,-4,1],[0,1,0]]),
            }
            fig, axes = plt.subplots(1, 5, figsize=(14, 3))
            axes[0].imshow(imagen, cmap="gray"); axes[0].set_title("entrada")
            for ax, (nombre, kernel) in zip(axes[1:4], kernels.items()):
                mapa = correlación_2d(imagen, kernel, padding=1)
                ax.imshow(mapa, cmap="coolwarm"); ax.set_title(nombre)
            pooled = max_pool_2d(imagen)
            axes[4].imshow(pooled, cmap="gray"); axes[4].set_title("max pooling")
            for ax in axes: ax.axis("off")
            plt.show()
            """
        ),
        md(
            r"""
            ## Varios canales y parámetros

            Un filtro 3D abarca todos los canales de entrada. Con $C_{in}$ canales,
            $C_{out}$ filtros y kernel $K_h\times K_w$, hay
            $K_hK_wC_{in}C_{out}+C_{out}$ parámetros: no depende del tamaño espacial.
            Pooling no aprende pesos; reduce resolución y aumenta el campo receptivo.
            """
        ),
        code(
            """
            def parámetros_conv(kh, kw, canales_entrada, filtros, bias=True):
                return kh*kw*canales_entrada*filtros + (filtros if bias else 0)

            print("Conv 3×3, 3→64:", parámetros_conv(3,3,3,64), "parámetros")
            print("Capa densa 224×224×3→64:", 224*224*3*64+64, "parámetros")
            """
        ),
        md(
            """
            **Ejercicios:** compare correlación y convolución rotando el kernel;
            implemente average pooling; calcule tamaños para stride 2; aplique los
            filtros a una matriz que represente un campo escalar y discuta relación
            con operadores diferenciales discretos.
            """
        ),
    ]
    write_notebook(path, cells)


def build_lenet_alexnet() -> None:
    path = "04_vision/41_lenet_alexnet.ipynb"
    cells = [
        header(path, "LeNet-5 y AlexNet", "¿Qué cambió al pasar de dígitos pequeños a visión profunda?", 4),
        md(
            """
            **Requiere PyTorch.** LeNet-5 fijó el patrón convolución–submuestreo–capa
            densa. AlexNet escaló profundidad, datos y cómputo, usando ReLU, GPU,
            augmentación y dropout. Entrenaremos una adaptación de LeNet sobre los
            dígitos 8×8 incluidos en scikit-learn, reescalados a 32×32. AlexNet se
            inspecciona sin entrenarlo para no convertir la clase en una espera.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn.functional as F
            from sklearn.datasets import load_digits
            from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
            from sklearn.model_selection import train_test_split
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
            from torchvision.models import alexnet

            SEMILLA = 42
            torch.manual_seed(SEMILLA)
            dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            datos = load_digits()
            X = torch.tensor(datos.images[:,None]/16.0, dtype=torch.float32)
            X = F.interpolate(X, size=(32,32), mode="bilinear", align_corners=False)
            y = torch.tensor(datos.target, dtype=torch.long)
            índices = np.arange(len(y))
            dev, test = train_test_split(índices, test_size=.2, stratify=y, random_state=SEMILLA)
            train, val = train_test_split(dev, test_size=.2, stratify=y[dev], random_state=SEMILLA)
            loader_train = DataLoader(TensorDataset(X[train],y[train]), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEMILLA))
            loader_val = DataLoader(TensorDataset(X[val],y[val]), batch_size=256)
            """
        ),
        code(
            """
            class LeNet5(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 6, kernel_size=5), nn.Tanh(), nn.AvgPool2d(2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(), nn.AvgPool2d(2),
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(), nn.Linear(16*5*5,120), nn.Tanh(),
                        nn.Linear(120,84), nn.Tanh(), nn.Linear(84,10),
                    )
                def forward(self,x): return self.classifier(self.features(x))

            modelo = LeNet5().to(dispositivo)
            criterio = nn.CrossEntropyLoss()
            optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
            for época in range(12):
                modelo.train()
                for xb,yb in loader_train:
                    xb,yb = xb.to(dispositivo),yb.to(dispositivo)
                    optimizador.zero_grad(); pérdida=criterio(modelo(xb),yb)
                    pérdida.backward(); optimizador.step()
                modelo.eval()
                with torch.no_grad():
                    aciertos=sum((modelo(xb.to(dispositivo)).argmax(1).cpu()==yb).sum().item() for xb,yb in loader_val)
                if época in [0,3,7,11]: print(época+1, aciertos/len(val))
            """
        ),
        code(
            """
            modelo.eval()
            with torch.no_grad(): pred=modelo(X[test].to(dispositivo)).argmax(1).cpu().numpy()
            print("accuracy test:", accuracy_score(y[test],pred))
            ConfusionMatrixDisplay.from_predictions(y[test],pred,cmap="Blues")
            plt.show()

            alex = alexnet(weights=None)
            print("Parámetros LeNet:", sum(p.numel() for p in modelo.parameters()))
            print("Parámetros AlexNet:", sum(p.numel() for p in alex.parameters()))
            print("Salida AlexNet para 2 imágenes RGB:", alex(torch.randn(2,3,224,224)).shape)
            del alex
            """
        ),
        md(
            """
            **Ejercicios:** cambie tanh/average pooling por ReLU/max pooling; haga una
            ablación justa; calcule manualmente cada tamaño; explique por qué ImageNet
            y disponibilidad de GPU fueron tan importantes como la arquitectura.
            """
        ),
    ]
    write_notebook(path, cells, tier="pytorch")


def build_architecture_zoo() -> None:
    path = "04_vision/42_vgg_resnet_inception_fractalnet.ipynb"
    cells = [
        header(path, "VGG, ResNet, Inception y FractalNet", "¿Cómo permiten los caminos de información entrenar redes profundas?", 4),
        md(
            """
            **Requiere TensorFlow.** ImageNet es un dataset y una competición, no una
            arquitectura. VGG apila convoluciones 3×3; Inception procesa varias
            escalas en paralelo; ResNet aprende un residuo $F(x)$ y suma $x+F(x)$;
            FractalNet construye múltiples caminos auto-similares. Aquí inspeccionamos
            bloques y tamaños; el notebook siguiente aborda transferencia.
            """
        ),
        code(
            """
            import gc
            import pandas as pd
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            print("TensorFlow", tf.__version__)

            def bloque_residual(x, filtros):
                atajo = x
                y = layers.Conv2D(filtros,3,padding="same",use_bias=False)(x)
                y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
                y = layers.Conv2D(filtros,3,padding="same",use_bias=False)(y)
                y = layers.BatchNormalization()(y)
                if x.shape[-1] != filtros: atajo = layers.Conv2D(filtros,1)(atajo)
                return layers.ReLU()(layers.Add()([atajo,y]))

            def bloque_inception(x, filtros=16):
                r1 = layers.Conv2D(filtros,1,padding="same",activation="relu")(x)
                r3 = layers.Conv2D(filtros,3,padding="same",activation="relu")(x)
                r5 = layers.Conv2D(filtros,5,padding="same",activation="relu")(x)
                rp = layers.MaxPool2D(3,strides=1,padding="same")(x)
                rp = layers.Conv2D(filtros,1,activation="relu")(rp)
                return layers.Concatenate()([r1,r3,r5,rp])

            def bloque_fractal(x, filtros, profundidad):
                directo = layers.Conv2D(filtros,3,padding="same",activation="relu")(x)
                if profundidad == 1: return directo
                largo = bloque_fractal(x,filtros,profundidad-1)
                largo = bloque_fractal(largo,filtros,profundidad-1)
                return layers.Average()([directo,largo])
            """
        ),
        code(
            """
            entrada = keras.Input((32,32,16))
            bloques = {
                "residual": bloque_residual(entrada,16),
                "inception": bloque_inception(entrada,16),
                "fractal_C3": bloque_fractal(entrada,16,3),
            }
            for nombre,salida in bloques.items():
                modelo = keras.Model(entrada,salida,name=nombre)
                print(nombre, "salida", modelo.output_shape, "parámetros", modelo.count_params())
            """
        ),
        code(
            """
            fábricas = {
                "VGG16": lambda: keras.applications.VGG16(weights=None,include_top=False,input_shape=(96,96,3)),
                "ResNet50": lambda: keras.applications.ResNet50(weights=None,include_top=False,input_shape=(96,96,3)),
                "InceptionV3": lambda: keras.applications.InceptionV3(weights=None,include_top=False,input_shape=(96,96,3)),
            }
            filas=[]
            for nombre,fábrica in fábricas.items():
                keras.backend.clear_session(); modelo=fábrica()
                salida=modelo(tf.zeros((1,96,96,3)),training=False)
                filas.append({"arquitectura":nombre,"parámetros":modelo.count_params(),"salida":str(tuple(salida.shape))})
                del modelo; gc.collect()
            display(pd.DataFrame(filas).set_index("arquitectura"))
            """
        ),
        md(
            """
            **Preguntas:** ¿por qué $x+F(x)$ ofrece una ruta de gradiente? ¿Qué costo
            tiene concatenar ramas Inception? ¿En qué se distingue FractalNet de una
            simple red residual? Compare número de parámetros, activaciones y FLOPs:
            parámetros no son una medida suficiente de costo.
            """
        ),
    ]
    write_notebook(path, cells, tier="tensorflow")


def build_transfer_learning() -> None:
    path = "04_vision/43_transfer_learning_fisica.ipynb"
    cells = [
        header(path, "Transfer learning con modelos ImageNet", "¿Cuándo reutilizar representaciones visuales ayuda a un problema nuevo?", 4),
        md(
            """
            **Requiere TensorFlow e Internet para descargar CIFAR-10 y pesos.** El
            flujo tiene dos fases: congelar una base preentrenada y ajustar la cabeza;
            luego, opcionalmente, descongelar pocas capas con tasa muy pequeña. El
            test se mantiene cerrado. MobileNetV2 permite una práctica rápida; cambie
            una constante para usar InceptionV3.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import tensorflow as tf
            from sklearn.metrics import accuracy_score
            from tensorflow import keras
            from tensorflow.keras import layers

            SEMILLA=42
            keras.utils.set_random_seed(SEMILLA)
            ARQUITECTURA="MobileNetV2"  # cambie a "InceptionV3" para el laboratorio extendido
            (X_train,y_train),(X_test,y_test)=keras.datasets.cifar10.load_data()
            y_train,y_test=y_train.ravel(),y_test.ravel()
            clases=[0,1,2]  # avión, automóvil, ave: ejemplo pequeño
            mask_train=np.isin(y_train,clases); mask_test=np.isin(y_test,clases)
            X_dev,y_dev=X_train[mask_train][:4500],y_train[mask_train][:4500]
            X_test,y_test=X_test[mask_test][:1200],y_test[mask_test][:1200]
            orden=np.random.default_rng(SEMILLA).permutation(len(X_dev))
            corte=int(.8*len(orden)); tr,val=orden[:corte],orden[corte:]
            """
        ),
        code(
            """
            if ARQUITECTURA=="InceptionV3":
                tamaño=(96,96); Base=keras.applications.InceptionV3; preprocesar=keras.applications.inception_v3.preprocess_input
            else:
                tamaño=(96,96); Base=keras.applications.MobileNetV2; preprocesar=keras.applications.mobilenet_v2.preprocess_input

            aumentación=keras.Sequential([
                layers.RandomFlip("horizontal"), layers.RandomRotation(.05),
            ],name="aumentación")
            base=Base(weights="imagenet",include_top=False,input_shape=(*tamaño,3))
            base.trainable=False
            entrada=keras.Input((32,32,3))
            x=layers.Resizing(*tamaño)(entrada); x=aumentación(x); x=preprocesar(x)
            x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(.25)(x)
            salida=layers.Dense(len(clases))(x)
            modelo=keras.Model(entrada,salida)
            modelo.compile(optimizer=keras.optimizers.Adam(1e-3),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
            modelo.fit(X_dev[tr],y_dev[tr],validation_data=(X_dev[val],y_dev[val]),epochs=5,batch_size=64,verbose=2,
                       callbacks=[keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)])
            """
        ),
        code(
            """
            # Fine-tuning: sólo las últimas capas y con tasa 100 veces menor.
            base.trainable=True
            for capa in base.layers[:-20]: capa.trainable=False
            modelo.compile(optimizer=keras.optimizers.Adam(1e-5),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
            modelo.fit(X_dev[tr],y_dev[tr],validation_data=(X_dev[val],y_dev[val]),epochs=3,batch_size=64,verbose=2)
            pred=modelo.predict(X_test,batch_size=128,verbose=0).argmax(1)
            print("accuracy test final:",accuracy_score(y_test,pred))
            """
        ),
        md(
            """
            **Aplicación física propuesta:** sustituya CIFAR por imágenes de lentes
            gravitacionales, cámaras de niebla o microscopía con licencia explícita.
            Compare desde cero, extractor congelado y fine-tuning; controle que un
            mismo objeto/experimento no aparezca en particiones distintas. Discuta el
            cambio de dominio: las texturas de ImageNet no son leyes físicas.
            """
        ),
    ]
    write_notebook(path, cells, tier="tensorflow-network", accelerator="gpu")


def build_autoencoder() -> None:
    path = "05_generativos/50_autoencoders.ipynb"
    cells = [
        header(path, "Autoencoders", "¿Qué representación comprimida conserva la información útil?", 4),
        md(
            r"""
            **Requiere TensorFlow.** Un encoder $z=f_\theta(x)$ y un decoder
            $\hat x=g_\phi(z)$ minimizan reconstrucción. Un cuello de botella fuerza
            compresión, pero no garantiza variables físicamente interpretables. La
            detección de anomalías presupone que el entrenamiento representa la
            normalidad.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import tensorflow as tf
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split
            from tensorflow import keras
            from tensorflow.keras import layers

            keras.utils.set_random_seed(42)
            X=load_digits().data.astype("float32")/16.0
            X_dev,X_test=train_test_split(X,test_size=.2,random_state=42)
            X_train,X_val=train_test_split(X_dev,test_size=.2,random_state=42)
            entrada=keras.Input((64,))
            z=layers.Dense(32,activation="relu")(entrada); z=layers.Dense(8,name="latente")(z)
            x=layers.Dense(32,activation="relu")(z); salida=layers.Dense(64,activation="sigmoid")(x)
            autoencoder=keras.Model(entrada,salida)
            encoder=keras.Model(entrada,z)
            autoencoder.compile(optimizer="adam",loss="mse")
            historia=autoencoder.fit(X_train,X_train,validation_data=(X_val,X_val),epochs=80,batch_size=64,verbose=0,
                                     callbacks=[keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True)])
            print("dimensión",X.shape[1],"→",encoder.output_shape[-1],"| test MSE",autoencoder.evaluate(X_test,X_test,verbose=0))
            """
        ),
        code(
            """
            reconstruidas=autoencoder.predict(X_test[:10],verbose=0)
            fig,axes=plt.subplots(2,10,figsize=(14,3))
            for i in range(10):
                axes[0,i].imshow(X_test[i].reshape(8,8),cmap="gray"); axes[1,i].imshow(reconstruidas[i].reshape(8,8),cmap="gray")
                axes[0,i].axis("off"); axes[1,i].axis("off")
            axes[0,0].set_ylabel("original"); axes[1,0].set_ylabel("reconstrucción")
            plt.show()

            rng=np.random.default_rng(42)
            anomalías=np.clip(X_test+rng.normal(0,.35,X_test.shape),0,1)
            err_normal=np.mean((X_test-autoencoder.predict(X_test,verbose=0))**2,axis=1)
            err_anómalo=np.mean((anomalías-autoencoder.predict(anomalías,verbose=0))**2,axis=1)
            plt.hist(err_normal,alpha=.6,label="normal"); plt.hist(err_anómalo,alpha=.6,label="perturbado"); plt.legend(); plt.xlabel("error"); plt.show()
            """
        ),
        md(
            """
            **Ejercicios:** compare dimensión latente 2, 8 y 32; visualice el espacio
            2D; entrene denoising autoencoder; establezca un umbral sólo con
            validación; explique por qué una anomalía bien reconstruida puede escapar.
            """
        ),
    ]
    write_notebook(path, cells, tier="tensorflow")


def build_gan() -> None:
    path = "05_generativos/51_gan.ipynb"
    cells = [
        header(path, "Redes generativas adversarias", "¿Cómo aprende un generador mediante un adversario?", 4),
        md(
            r"""
            **Requiere PyTorch; GPU opcional.** El juego clásico es
            $\min_G\max_D\;E_{x\sim p_d}\log D(x)+E_z\log(1-D(G(z)))$.
            Para el generador usamos la pérdida no saturante $-\log D(G(z))$.
            Entrenaremos sobre dígitos 8×8: es un laboratorio de dinámica, no un
            generador de alta fidelidad.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            from sklearn.datasets import load_digits
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            SEMILLA=42; torch.manual_seed(SEMILLA)
            dispositivo=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X=torch.tensor(load_digits().data/16.0,dtype=torch.float32)
            loader=DataLoader(TensorDataset(X),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(SEMILLA),drop_last=True)
            G=nn.Sequential(nn.Linear(16,64),nn.LeakyReLU(.2),nn.Linear(64,128),nn.LeakyReLU(.2),nn.Linear(128,64),nn.Sigmoid()).to(dispositivo)
            D=nn.Sequential(nn.Linear(64,128),nn.LeakyReLU(.2),nn.Dropout(.2),nn.Linear(128,64),nn.LeakyReLU(.2),nn.Linear(64,1)).to(dispositivo)
            optG=torch.optim.Adam(G.parameters(),lr=2e-4,betas=(.5,.999)); optD=torch.optim.Adam(D.parameters(),lr=2e-4,betas=(.5,.999))
            bce=nn.BCEWithLogitsLoss(); historia=[]
            """
        ),
        code(
            """
            for época in range(120):
                for (real,) in loader:
                    real=real.to(dispositivo); n=len(real)
                    z=torch.randn(n,16,device=dispositivo); falso=G(z)
                    optD.zero_grad()
                    lossD=bce(D(real),torch.ones(n,1,device=dispositivo))+bce(D(falso.detach()),torch.zeros(n,1,device=dispositivo))
                    lossD.backward(); optD.step()
                    z=torch.randn(n,16,device=dispositivo)
                    optG.zero_grad(); lossG=bce(D(G(z)),torch.ones(n,1,device=dispositivo)); lossG.backward(); optG.step()
                historia.append((lossD.item(),lossG.item()))
                if época%30==0: print(época, historia[-1])
            """
        ),
        code(
            """
            hist=np.asarray(historia)
            plt.plot(hist[:,0],label="D"); plt.plot(hist[:,1],label="G"); plt.legend(); plt.xlabel("época"); plt.show()
            with torch.no_grad(): muestras=G(torch.randn(20,16,device=dispositivo)).cpu().reshape(-1,8,8)
            fig,axes=plt.subplots(4,5,figsize=(7,6))
            for ax,img in zip(axes.flat,muestras): ax.imshow(img,cmap="gray"); ax.axis("off")
            plt.show()
            """
        ),
        md(
            """
            Las pérdidas no son una métrica suficiente: un generador puede colapsar
            a pocos modos. **Ejercicios:** guarde muestras cada 10 épocas; mida
            diversidad entre pares; alterne más pasos de D; compare ruido latente;
            investigue WGAN conceptualmente y explique qué cambia en la distancia.
            """
        ),
    ]
    write_notebook(path, cells, tier="pytorch", accelerator="gpu-optional")


def build_rnn_lstm_gru() -> None:
    path = "06_secuencias/60_rnn_lstm_gru.ipynb"
    cells = [
        header(path, "RNN, LSTM y GRU", "¿Cómo conserva una red información temporal?", 4),
        md(
            r"""
            **Requiere TensorFlow.** Una RNN actualiza
            $h_t=\phi(W_xx_t+W_hh_{t-1}+b)$. El gradiente multiplica repetidamente
            Jacobianos y puede desaparecer o explotar. LSTM introduce celda y
            compuertas; GRU combina compuertas con menos parámetros. Compararemos las
            tres en predicción de señales amortiguadas.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            SEMILLA=42; keras.utils.set_random_seed(SEMILLA); rng=np.random.default_rng(SEMILLA)
            n=1800; longitud=60
            t=np.linspace(0,6,longitud+1)
            X=[]; y=[]
            for _ in range(n):
                w=rng.uniform(.8,2.5); gamma=rng.uniform(.02,.25); fase=rng.uniform(0,2*np.pi)
                señal=np.exp(-gamma*t)*np.sin(w*t+fase)+rng.normal(0,.02,len(t))
                X.append(señal[:-1,None]); y.append(señal[-1])
            X=np.asarray(X,dtype="float32"); y=np.asarray(y,dtype="float32")
            orden=rng.permutation(n); train,val,test=orden[:1200],orden[1200:1500],orden[1500:]
            """
        ),
        code(
            """
            capas={"SimpleRNN":layers.SimpleRNN,"LSTM":layers.LSTM,"GRU":layers.GRU}
            filas=[]; modelos={}
            for nombre,Capa in capas.items():
                keras.backend.clear_session(); keras.utils.set_random_seed(SEMILLA)
                modelo=keras.Sequential([layers.Input((longitud,1)),Capa(24),layers.Dense(1)])
                modelo.compile(optimizer=keras.optimizers.Adam(1e-3),loss="mse")
                hist=modelo.fit(X[train],y[train],validation_data=(X[val],y[val]),epochs=35,batch_size=64,verbose=0,
                                callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
                mse=modelo.evaluate(X[test],y[test],verbose=0)
                filas.append({"modelo":nombre,"parámetros":modelo.count_params(),"MSE_test":mse,"épocas":len(hist.history["loss"])})
                modelos[nombre]=modelo
            display(pd.DataFrame(filas).set_index("modelo"))
            """
        ),
        code(
            """
            mejor=min(filas,key=lambda f:f["MSE_test"])["modelo"]
            pred=modelos[mejor].predict(X[test],verbose=0).ravel()
            plt.scatter(y[test],pred,alpha=.5); límites=[min(y[test].min(),pred.min()),max(y[test].max(),pred.max())]
            plt.plot(límites,límites,"k--"); plt.xlabel("siguiente valor real"); plt.ylabel("predicción"); plt.title(mejor); plt.show()
            """
        ),
        md(
            r"""
            **Ejercicios:** aumente longitud sin cambiar unidades; registre norma de
            gradiente; compare contra persistencia $\hat x_{t+1}=x_t$; prediga varios
            pasos de forma autoregresiva y observe acumulación de error.
            """
        ),
    ]
    write_notebook(path, cells, tier="tensorflow")


def build_seq2seq() -> None:
    path = "06_secuencias/61_seq2seq_atencion.ipynb"
    cells = [
        header(path, "Seq2Seq con atención", "¿Cómo aprende un decoder qué partes de la entrada consultar?", 4),
        md(
            r"""
            **Requiere PyTorch.** Un encoder produce estados $h_1,\ldots,h_T$. En el
            paso $s$, atención calcula $\alpha_{st}=\mathrm{softmax}(q_s^Th_t)$ y
            contexto $c_s=\sum_t\alpha_{st}h_t$. Entrenaremos una tarea transparente:
            invertir secuencias de dígitos. Los mapas de atención tienen una respuesta
            esperada: una diagonal invertida.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            from torch import nn

            SEMILLA=42; torch.manual_seed(SEMILLA); rng=np.random.default_rng(SEMILLA)
            dispositivo=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            VOCAB=11; SOS=10; L=6; N=3000
            fuente=torch.tensor(rng.integers(0,10,size=(N,L)),dtype=torch.long)
            objetivo=torch.flip(fuente,dims=[1])
            train,val,test=torch.arange(0,2200),torch.arange(2200,2600),torch.arange(2600,N)

            class Seq2SeqAtención(nn.Module):
                def __init__(self,emb=24,oculto=48):
                    super().__init__(); self.embed=nn.Embedding(VOCAB,emb); self.encoder=nn.GRU(emb,oculto,batch_first=True)
                    self.decoder=nn.GRUCell(emb+oculto,oculto); self.salida=nn.Linear(2*oculto,10)
                def forward(self,x,target=None):
                    enc,h=self.encoder(self.embed(x)); h=h.squeeze(0); previo=torch.full((len(x),),SOS,device=x.device)
                    logits=[]; atenciones=[]
                    for s in range(L):
                        pesos=torch.softmax(torch.bmm(enc,h.unsqueeze(2)).squeeze(2),dim=1)
                        contexto=torch.bmm(pesos.unsqueeze(1),enc).squeeze(1)
                        h=self.decoder(torch.cat([self.embed(previo),contexto],dim=1),h)
                        logit=self.salida(torch.cat([h,contexto],dim=1)); logits.append(logit); atenciones.append(pesos)
                        previo=target[:,s] if target is not None else logit.argmax(1)
                    return torch.stack(logits,dim=1),torch.stack(atenciones,dim=1)
            """
        ),
        code(
            """
            modelo=Seq2SeqAtención().to(dispositivo); opt=torch.optim.Adam(modelo.parameters(),lr=2e-3); ce=nn.CrossEntropyLoss()
            for época in range(35):
                perm=train[torch.randperm(len(train))]
                modelo.train()
                for inicio in range(0,len(perm),128):
                    idx=perm[inicio:inicio+128]; xb=fuente[idx].to(dispositivo); yb=objetivo[idx].to(dispositivo)
                    opt.zero_grad(); logits,_=modelo(xb,yb); loss=ce(logits.reshape(-1,10),yb.reshape(-1)); loss.backward()
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(),1.0); opt.step()
                if época%10==0:
                    modelo.eval()
                    with torch.no_grad(): pred=modelo(fuente[val].to(dispositivo))[0].argmax(2).cpu()
                    print(época,"secuencias exactas",(pred==objetivo[val]).all(1).float().mean().item())
            """
        ),
        code(
            """
            modelo.eval()
            with torch.no_grad(): logits,A=modelo(fuente[test[:1]].to(dispositivo)); pred=logits.argmax(2).cpu()
            print("entrada",fuente[test[0]].tolist(),"objetivo",objetivo[test[0]].tolist(),"pred",pred[0].tolist())
            plt.imshow(A[0].cpu(),cmap="viridis",vmin=0,vmax=1); plt.xlabel("posición de entrada"); plt.ylabel("paso de salida"); plt.colorbar(label="atención"); plt.show()
            """
        ),
        md(
            """
            **Ejercicios:** quite atención y use sólo el último estado; aumente la
            longitud; reemplace teacher forcing total por probabilidad decreciente;
            explique por qué un mapa de atención no es automáticamente una explicación
            causal de la decisión.
            """
        ),
    ]
    write_notebook(path, cells, tier="pytorch")


def build_physics_timeseries() -> None:
    path = "06_secuencias/62_series_temporales_fisicas.ipynb"
    cells = [
        header(path, "Series temporales físicas: sistema de Lorenz", "¿Cómo evaluamos predicción cuando el tiempo impide mezclar observaciones?", 4),
        md(
            r"""
            **Requiere TensorFlow.** Integramos Lorenz y predecimos $x_{t+1}$ desde
            una ventana de $(x,y,z)$. La separación es cronológica: pasado para
            entrenamiento, intervalo posterior para validación y futuro para test.
            En un sistema caótico, error pequeño de un paso no implica trayectoria
            correcta a largo plazo.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            from scipy.integrate import solve_ivp
            from sklearn.linear_model import Ridge
            from sklearn.metrics import root_mean_squared_error
            from sklearn.preprocessing import StandardScaler
            from tensorflow import keras
            from tensorflow.keras import layers

            def lorenz(t,s,σ=10,ρ=28,β=8/3):
                x,y,z=s; return [σ*(y-x),x*(ρ-z)-y,x*y-β*z]
            t=np.linspace(0,80,8001); señal=solve_ivp(lorenz,[t[0],t[-1]],[1,1,1],t_eval=t,rtol=1e-9,atol=1e-9).y.T[1000:]
            n_train=int(.65*len(señal)); n_val=int(.82*len(señal))
            scaler=StandardScaler().fit(señal[:n_train]); S=scaler.transform(señal)
            ventana=40
            X=np.array([S[i:i+ventana] for i in range(len(S)-ventana)])
            y=np.array([S[i+ventana,0] for i in range(len(S)-ventana)])
            idx=np.arange(len(X)); train=idx[idx+ventana<n_train]; val=idx[(idx+ventana>=n_train)&(idx+ventana<n_val)]; test=idx[idx+ventana>=n_val]
            """
        ),
        code(
            """
            persistencia=X[test,-1,0]
            ridge=Ridge(alpha=1.0).fit(X[train].reshape(len(train),-1),y[train])
            pred_ridge=ridge.predict(X[test].reshape(len(test),-1))
            keras.utils.set_random_seed(42)
            modelo=keras.Sequential([layers.Input((ventana,3)),layers.LSTM(32),layers.Dense(1)])
            modelo.compile(optimizer="adam",loss="mse")
            modelo.fit(X[train],y[train],validation_data=(X[val],y[val]),epochs=30,batch_size=128,shuffle=False,verbose=0,
                       callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
            pred_lstm=modelo.predict(X[test],verbose=0).ravel()
            display(pd.DataFrame({"RMSE":[root_mean_squared_error(y[test],persistencia),root_mean_squared_error(y[test],pred_ridge),root_mean_squared_error(y[test],pred_lstm)]},index=["persistencia","Ridge","LSTM"]))
            """
        ),
        code(
            """
            tramo=slice(0,500)
            plt.figure(figsize=(12,4)); plt.plot(y[test][tramo],label="real"); plt.plot(pred_lstm[tramo],label="LSTM",alpha=.8); plt.plot(persistencia[tramo],label="persistencia",alpha=.6); plt.legend(); plt.ylabel("x estandarizado"); plt.show()
            """
        ),
        md(
            """
            **Ejercicios:** compare predicción directa a 10 pasos; genere rollout
            autoregresivo; mida horizonte hasta que error supere una desviación;
            compare con el tiempo de Lyapunov; explique por qué barajar ventanas antes
            de separar produce una evaluación optimista.
            """
        ),
    ]
    write_notebook(path, cells, tier="tensorflow")


if __name__ == "__main__":
    build_convolution_numpy()
    build_lenet_alexnet()
    build_architecture_zoo()
    build_transfer_learning()
    build_autoencoder()
    build_gan()
    build_rnn_lstm_gru()
    build_seq2seq()
    build_physics_timeseries()
    print("Notebooks de arquitecturas profundas generados.")
