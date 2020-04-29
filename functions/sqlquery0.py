import os
import sqlite3
import pandas as pd
import psycopg2

# Create a database for local environment
conn = sqlite3.connect('flow-ez.db')
conn = sqlite3.connect('flow-ez.db', check_same_thread=False) ## Important

# Create a database for local Heroku
conn.row_factory = sqlite3.Row
# cursor = conn.cursor()

# conn.row_factory = sqlite3.Row

# Create Huroku remote DB

# DATABASE_URL = os.environ['DATABASE_URL']
# print('DB URL: \n', DATABASE_URL)
# conn = psycopg2.connect(DATABASE_URL)

# Make a convenience function for running SQL queries
def sql_query(query):
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
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
