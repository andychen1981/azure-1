from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

db = SQLAlchemy(app)
from app import db

def init_db():
    import models
    Base.metadata.create_all(bind=engine)

Base = declarative_base()

class Data_Table(db.Model):

    __tablename__ = "data_table"

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String, nullable=False)
    last_name = db.Column(db.String, nullable=False)
    mea_date = db.Column(db.String, nullable=False)
    disp_date = db.Column(db.String, nullable=False)
    time_1 = db.Column(db.DateTime, nullable=False)
    dev_id = db.Column(db.String, nullable=False)
    qr_code = db.Column(db.String, nullable=False)
    loc = db.Column(db.String, nullable=False)
    pulse = db.Column(db.String, nullable=False)
    fft = db.Column(db.String, nullable=False)
    trend = db.Column(db.String, nullable=False)
    res = db.Column(db.String, nullable=False)
    prob = db.Column(db.Integer, nullable=False)
    msg = db.Column(db.String, nullable=False)

    def __repr__(self):
        return '<Data {}'.format(self.name)



