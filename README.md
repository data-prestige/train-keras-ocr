# how to train keras-ocr 

Per il training del OCR è stata utilizzata la seguente libreria per la quale è riportata di seguito la documentazione
https://keras-ocr.readthedocs.io/en/latest/examples/fine_tuning_recognizer.html

In questa repository è contenuto il codice per un retraining del modello CRNN per il Plate Recognition.
Il retraining è effettuato con un vocabolario di 36 classi. Le lettere maiuscole dell'alfabeto inglese da A a Z e i numeri da 0 a 9.
Il vocabolario è contenuto nel file vocabolary.py. Laddove sussita l'esigenza di eliminare o aggiungere simboli si può intervenire direttamente all'intero del file.

## Riallenare il modello con il dataset Utilizzato per il Plate Recognition OCR

All'interno della repository sono contenuti dei samples per avviare il retraining, ma visto il peso dei dataset questi verranno scaricati a runtime con i comandi gdown.

1) Aprire il notebook denominato end-to-end-train.ipynb (con colab o jupyter)
2) Eseguire in succesione le celle. A questo punto partirà l'esecuzione e si vedrà l'avanzamento dello epoche di training con velocità variabile in base alle risorse (RAM, GPU, CPU) della macchina
3) Il training si fermerà quando la loss smette di migliorare entro un certa range
4) A questo punto si troverà il file dei pesi riallenati all'interno della cartella "train-keras-ocr" con nome "recognizer_borndigital.h5"

Per eseguire delle predizioni di test si possono utilizzare i samples contenuti nella directory "test" eseguendo la cella denominata "#test di predizione"
Se si voglio caricare dei file custom per la predizione basta caricarli nella cartella "test" avendoli prima rinominati con il valore della targa da predirre.
Al termine della predizione verrà stampato il valore di accuratezza e il numero di samples correttamente predetti
