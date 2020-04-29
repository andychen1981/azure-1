import os
import sqlite3
import pandas as pd
import psycopg2

# Create a database for local environment
# conn = sqlite3.connect('flow-ez.db')
# conn = sqlite3.connect('flow-ez.db', check_same_thread=False) ## Important
    # cursor = conn.cursor()

# conn.row_factory = sqlite3.Row

# Create Huroku remote DB


connection = psycopg2.connect(user = "csefwzficaoouh",
                                password = "4bef0ab168c67e5aeebb8152e3de4995e5cb733268609c5b13d42348a51dd8f3",
                                host = "ec2-174-129-254-217.compute-1.amazonaws.com",
                                port = "5432",
                                database = "d30b3p3ckp94hl")

DATABASE_URL = os.environ['DATABASE_URL']
print('DB URL: ', DATABASE_URL)
conn = psycopg2.connect(DATABASE_URL)

# Make a convenience function for running SQL queries
def sql_query(query):
    cur = conn.cursor()
    cur.execute(query)
    # rows = cur.fetchall()
    rows = [dict(first_name=row[0],last_name=row[1],mea_date=row[2],disp_date=row[3],time_1=row[4],dev_id=row[5], qr_code=row[6],loc=row[7],res=row[8],prob=row[9]) for row in cur.fetchall()]
    return rows
    conn.commit() #Andy

def sql_edit_insert(query,var):
    cur = conn.cursor()
    cur.execute(query,var)
    conn.commit()

def sql_delete(query,var):
    cur = conn.cursor()
    cur.execute(query,var)
    conn.commit()
    
def sql_query2(query,var):
    cur = conn.cursor()
    cur.execute(query,var)
    rows = cur.fetchall()
    return rows
    conn.commit() #Andy 
