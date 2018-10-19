## Descrição

Esse projeto busca utilizar o conceito de *Bag of Visual Words* com uma *Comparação de Histogramas* para encontrar imagens similares a uma imagem de entrada.

A partir de um dataset de imagens, é feito um vocabulário de *características* de imagens. O vocabulários é gerado usando o algortimo *KMeans* a partir de descritores das imagens. Os descritores da imagem são encontrados utilizando o algortimo *SURF*. Com esse vocabulário, pode-se gerar um histograma dos conteúdos do vocabulário para cada imagem, resultado em um dataset de histogramas das imagens. Esses histogramas e o vocabulário são armazenados em um arquivo para uso futuro.

O valor de similaridade entre imagens é encontrado a partir da comparação entre histogramas das imagens utilizando o algortimo *Chi Square*.

Portanto, a partir de uma imagem exemplo, o programa busca encontrar as imagens mais semelhantes ao exemplo no dataset, comparando o histograma do exemplo com os calculados previamante.

## Utilização

As imagens do dataset de treino devem estar dentro da pasta `images`. Elas podem estar em sub diretórios, são considerados todos arquivos presentes na pasta recursivamente.

Para criar um novo arquivo de histogramas do dataset e um novo vocabulário, execute o arquivo `create_hist.py`

```
$ python create_hist.py
```

Os histogramas são salvos no arquivo `hists.pkl` e o vocabulário no arquivo `vocab.pkl`

Para encontrar as imagens semelhantes a um exemplo, execute o arquivo `get_similar_images.py` trocando a variável `IMG_PATH` para o path da imagem desejada.

```
$ python get_similar_images.py
```