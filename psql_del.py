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
    query1 = "DELETE FROM data_table where first_name = 'Jimmy';"
    query2 = "DELETE FROM data_table where first_name = 'John';"
  
    cursor.execute(query1)
    cursor.execute(query2)
    
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