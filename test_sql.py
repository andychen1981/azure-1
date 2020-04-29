import sqlite3

conn=sqlite3.connect('/Users/andychen/myproject/sqlite.db')
c=conn.cursor()

# c.execute(""" create table pat2 (
# 	first text,
# 	last text
# 	)""")

# c.execute("insert into pat2 values ('one', 'two')")
# c.execute("select * from pat2 where last='two'")

# c.execute("insert into pat values ('null', 'DA', 'test')")
c.execute("select * from pat")
print(c.fetchall())
conn.commit()
conn.close()