# Data Science Full Stack Assignment di Gianluca Oldani


Il progetto è strutturato come segue:
1. main.py : script contenente il main() del codice. Richiama le funzionalità di recupero dei dati del database, di preparazione dei dati e plotting dei dati. 
All'interno del file sono anche implementati i modelli usati per fare la previsione del target.
2. package "data_preparation": contiene utility per la preparazione dei dati ed il loro plotting
3. package "db_connection": contiene utility per il recupero dei dati dal database.
4. Notebook_assignment_Gianluca_Oldani.ipynb: nootebook di Google Colab nel quale è spiegato il procedimento


Dipendenze del progetto:
* psycopg2 : driver per PostgreSQL
* matplotlib: libreria per grafici
* numpy: libreria per trattamento matrici dei dati ed utility operazioni matematiche
* statsmodels: libreria per la creazione dei modelli e summary delle statistiche dei modelli
* sklearn: libreria per il preprocessing dei dati, in questo caso usata per la standardizzazione
