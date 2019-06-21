# Search egnine
Este repositorio contiene un modelo base para generar el motor de búsqquedo con modelos predictivos

## Installation
Se debe utilizar python3

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

## Usage

```python
python3 main3.py  --path data.csv
python3 main.py  --path data.csv
```