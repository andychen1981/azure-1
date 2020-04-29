print('\n')
import psycopg2

try:
    connection = psycopg2.connect(user = "andy",
                                  password = " ",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "flow-ez")

    cursor = connection.cursor()

    create_table_query = '''CREATE TABLE data_table
         (first_name varchar(256),
          last_name   varchar(256),
          mea_date    varchar(14),
          disp_date   varchar(20),
          time_1      timestamp,  
          dev_id      varchar(12),
          qr_code     varchar(8),
          loc         varchar(2),
          pulse       varchar(128),
          fft         varchar(128),
          trend       varchar(128),
          res         varchar(8),
          prob        smallint,
          msg         varchar(36)); '''

    cursor.execute(create_table_query)
    connection.commit()
    print("Table created successfully in PostgreSQL ")
    query1 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Jimmy', 'Page','20191125100800','2019-11-01','003004802801','00000001','LO','Normal','50');"
    query2 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Robert', 'Plant','20191125100800','2019-11-02','003004802802','00000002','UP','Abnormal','90');"
    query3 = "insert into data_table (first_name, last_name, mea_date, disp_date, dev_id, qr_code, loc, res, prob) values ('Eric', 'Clapton','20191125100800','2019-11-03','003004802803','00000003','LO','Normal','30');"
    cursor.execute(query1)
    cursor.execute(query2)
    cursor.execute(query3)
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