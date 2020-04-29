print('\n')
import psycopg2

try:
    connection = psycopg2.connect(user = "andy",
                                  password = " ",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "flow-ez")

    cursor = connection.cursor()

    print("Table created successfully in PostgreSQL ")
    query1 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Jimmy', 'Jones','20191110100800','2019-11-09','003004802809','00000009','LO','Normal','55');"
    query2 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Robert', 'Ko','20191111100800','2019-11-10','003004802810','00000010','UP','Abnormal','95');"
    query3 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Johnson', 'Hsu','20191112100800','2019-11-11','003004802811','00000011','LO','Normal','35');"
    query4 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('John', 'Parr','20191113100800','2019-11-12','003004802812','00000012','LO','Abnormal','75');"
    cursor.execute(query1)
    cursor.execute(query2)
    cursor.execute(query3)
    cursor.execute(query4)
    connection.commit()

    postgreSQL_select_Query = "select * from data_table"

    cursor.execute(postgreSQL_select_Query)
    print("Selecting rows from patient data using cursor.fetchall")
    data_records = cursor.fetchall() 
   
    print("Print each row and it's columns values")
    for row in data_records:
       print("Patient = ", row[0], row[1],row[2],row[3], row[4],row[5],row[6], row[7],row[8],row[9])

except (Exception, psycopg2.Error) as error :
    print ("Error while fetching data from PostgreSQL", error)

finally:
    #closing database connection.
    if(connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")