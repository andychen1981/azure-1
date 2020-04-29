print('\n')
import psycopg2

try:
    connection = psycopg2.connect(user = "csefwzficaoouh",
                                password = "4bef0ab168c67e5aeebb8152e3de4995e5cb733268609c5b13d42348a51dd8f3",
                                host = "ec2-174-129-254-217.compute-1.amazonaws.com",
                                port = "5432",
                                database = "d30b3p3ckp94hl")

    cursor = connection.cursor()

    print("Table created successfully in PostgreSQL ")
    query1 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Jimmy', 'Jones','20191126100800','2019-11-05','003004802805','00000005','LO','Normal','55');"
    query2 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Robert', 'Ko','20191127100800','2019-11-06','003004802806','00000006','UP','Abnormal','95');"
    query3 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Johnson', 'Hsu','20191128100800','2019-11-07','003004802807','00000007','LO','Normal','35');"
    query4 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('John', 'Parr','20191129100800','2019-11-08','003004802808','00000008','LO','Abnormal','75');"
    # query5 = "UPDATE data_table set msg='NG' WHERE first_name='John' AND last_name='Lennon'"
             
    cursor.execute(query1)
    cursor.execute(query2)
    cursor.execute(query3)
    cursor.execute(query4)
    # cursor.execute(query5)
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