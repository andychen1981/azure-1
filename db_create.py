from app import db
from models import Pat

# create the database and the db table
db.create_all()

# insert data
db.session.add(Pat(dev_code='1234', qr_code='3456', log_id='1234', email='gmail', pwd='1234', pwd2='3456', name='Andy', nephro='John', sur='Mary', insur='Medicare', phone='1234', mea='today'))
# db.session.add(Pat('1264', '3456', '1234', '2gmail', '1234', '3456', '2Andy', 'John', '2Mary', 'Medicare', '1234', 'today'))
# db.session.add(Pat('1634', '3456', '1234', '3gmail', '1234', '3456', '3Andy', 'John', '3Mary', 'Medicare', '1234', 'today'))

# commit the changes
db.session.commit()