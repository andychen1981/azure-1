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
    query1 = "insert into mobile (ID, MODEL, PRICE) values (1, 'iPhone 6','399');"
    query2 = "insert into mobile (ID, MODEL, PRICE) values (2, 'Asus','299');"
    query3 = "insert into mobile (ID, MODEL, PRICE) values (3, 'Samsung','349');"
    cursor.execute(query1)
    cursor.execute(query2)
    cursor.execute(query3)

    postgreSQL_select_Query = "select * from mobile"

    cursor.execute(postgreSQL_select_Query)
    print("Selecting rows from mobile table using cursor.fetchall")
    data_records = cursor.fetchall() 
   
    print("Print each row and it's columns values")
    for row in data_records:
       print("First_Name = ", row[0])
       print("Last_Name = ", row[1])
       print("Mea_Date  = \n", row[2])

except (Exception, psycopg2.Error) as error :
    print ("Error while fetching data from PostgreSQL", error)

finally:
    #closing database connection.
    if(connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")