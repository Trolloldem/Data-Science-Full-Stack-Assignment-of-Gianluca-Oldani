import psycopg2

def get_data_from_db():
    try:

        #impostazione credenziali DB

        connect_str = {
            'dbname': '<DB_NAME>',
            'user': '<USERNAME>',
            'password': '<PASSWORD>',
            'host': '<HOST_URL>',
            'sslmode': '<SSLMODE>',
            'port': '<PORT>',
        }

        #creazione connessione al DB
        conn = psycopg2.connect(**connect_str)

        #creazione esecutore query
        cursor = conn.cursor()

        #query tabella variabile target
        cursor.execute("""SELECT * from <TABLE_NAME>""")
        conn.commit()

        rows_y = cursor.fetchall()

        #Stampa dello schema dei dati target
        print(cursor.description)

        # query tabella regressori
        cursor.execute("""SELECT * from <TABLE_NAME>""")
        conn.commit()

        rows_x = cursor.fetchall()

        # Stampa dello schema dei dati target
        print(cursor.description)

        #chiusura connessione al db e gestore query
        cursor.close()
        conn.close()

        return rows_x, rows_y

    except Exception as e:
        print(e)
