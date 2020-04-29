from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

from app import db

def init_db():
    import models
    Base.metadata.create_all(bind=engine)

Base = declarative_base()

class Pat(db.Model):

    __tablename__ = "pats"

    id = db.Column(db.Integer, primary_key=True)
    dev_code = db.Column(db.String, nullable=False)
    qr_code = db.Column(db.String, nullable=False)
    log_id = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False)
    pwd = db.Column(db.String, nullable=False)
    pwd2 = db.Column(db.String, nullable=False)
    name = db.Column(db.String, nullable=False)
    nephro = db.Column(db.String, nullable=False)
    sur = db.Column(db.String, nullable=False)
    insur = db.Column(db.String, nullable=False)
    phone = db.Column(db.String, nullable=False)
    mea = db.Column(db.String, nullable=False)

    def __repr__(self):
        return '<patient {}'.format(self.name)

class Nephro(db.Model):

    __tablename__ = "nephros"

    id = db.Column(db.Integer, primary_key=True)
    nephro_id = db.Column(db.String, nullable=False)
    clinic = db.Column(db.String, nullable=False)
    cont_pri = db.Column(db.String, nullable=False)
    addr = db.Column(db.String, nullable=False)
    phone = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False)
    sur_cont = db.Column(db.String, nullable=False)
    msg = db.Column(db.String, nullable=False)
    msg_stat = db.Column(db.String, nullable=False)
    treat = db.Column(db.String, nullable=False)

    def __repr__(self):
        return '<Nephrologist {}'.format(self.nephro)

class Sur(db.Model):

    __tablename__ = "surs"

    id = db.Column(db.Integer, primary_key=True)
    nephro_id = db.Column(db.String, nullable=False)
    sur_cont_id = db.Column(db.String, nullable=False)
    clinic = db.Column(db.String, nullable=False)
    cont_pri = db.Column(db.String, nullable=False)
    addr = db.Column(db.String, nullable=False)
    phone = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False)
    sched = db.Column(db.String, nullable=False)
    insur = db.Column(db.String, nullable=False)
    
    def __repr__(self):
        return '<Surgeon {}'.format(self.sur_id)

class Mea(db.Model):

    __tablename__ = "meas"

    id = db.Column(db.Integer, primary_key=True)
    mea_id = db.Column(db.String, nullable=False)
    mea_date = db.Column(db.String, nullable=False)
    loc = db.Column(db.String, nullable=False)
    sig = db.Column(db.String, nullable=False)
    res = db.Column(db.String, nullable=False)
    conf = db.Column(db.String, nullable=False)

class Treat(db.Model):

    __tablename__ = "treats"

    id = db.Column(db.Integer, primary_key=True)
    treat_id = db.Column(db.String, nullable=False)
    sur = db.Column(db.String, nullable=False)
    hos = db.Column(db.String, nullable=False)
    pre = db.Column(db.String, nullable=False)
    post = db.Column(db.String, nullable=False)

class Msg(db.Model):

    __tablename__ = "msgs"

    id = db.Column(db.Integer, primary_key=True)
    msg_id = db.Column(db.String, nullable=False)
    msg_stat = db.Column(db.String, nullable=False)
    msg_note = db.Column(db.String, nullable=False)
    
