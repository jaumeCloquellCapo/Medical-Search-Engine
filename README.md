# Medical Search Engine
Este repositorio contiene un ejemplo de modelo base para generar un motor de búsqueda con modelos predictivos

## Installation
Se debe utilizar python3 junto con estas librerias:

```bash
pip install sklearn
pip install nltk
pip install gensim
pip install panda
```

## Structure

### wordEmbeddings.py
 contiene el código para entrenar y predecir mediante word embedings (Word2Vect)

### tfIdf.py 
Contiene una función genérica para aplicar modelos supervisados y no supervisados (knn, svm, randomfporest ... )

### tokenizer.py: 
Contiene funciones genéricas de preprocesamiento y limpieza de los datos para posteriomente aplicar el modelo.

### data.csv: 
Contiene un dataset de especializaciones médicas para entrenar el modelo.


## Usage

```python
python3 main3.py  --path data.csv
python3 main.py  --path data.csv
```