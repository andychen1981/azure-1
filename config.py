import os

# default config
class BaseConfig(object):
    DEBUG = False
    # shortened for readability

    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    print ('Databse : 'SQLALCHEMY_DATABASE_URI)

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    print ('Databse : 'SQLALCHEMY_DATABASE_URI)


class ProductionConfig(BaseConfig):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    print ('Databse : 'SQLALCHEMY_DATABASE_URI)
